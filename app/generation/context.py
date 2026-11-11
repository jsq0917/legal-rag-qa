"""来源聚合：把问答库命中与检索命中合并为 prompt 参考资料 + SSE sources。

混合链路（§6.3）的上下文 = 结构化标准答案作"结论段" + 法条检索结果作"依据段"，
一次生成。本模块负责：

* 去重（同一 ``法名#条号`` 只保留最高分）；
* 排序（分数降序，进 prompt 与 sources 顺序一致）；
* 限量（``limit`` 条，超出丢弃）；
* 溯源字段（``law / chapter_path / article_no / score / snippet``）；
* prompt 上下文保留**完整正文**（QA 整段答案 / 法条父块全文），
  SSE sources 只带 snippet（前端引用标注够用即可）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain_profiles import profile_name_for_kb
from app.generation.prompts import ContextSource
from app.orchestration.qa_cache import QAPair
from app.security.guardrails import redact_sensitive

#: SSE sources 事件里 snippet 的默认长度
SNIPPET_CHARS = 120

#: 同类参考（未精确命中时的同话题问答）的固定相关度——
#: 无法条分数可比，统一置低，保证精确命中与法条命中排在前面
REFERENCE_SCORE = 0.1


@dataclass(frozen=True)
class SourceRef:
    """SSE ``sources`` 事件的一条（同时用于前端引用标注）。"""

    law: str
    chapter_path: str
    article_no: str
    score: float
    snippet: str
    chunk_id: str = ""
    kind: str = "law"  # law=法条 / qa=问答标准答案
    effective_from: str = ""
    effective_to: str = ""
    version: int = 0
    kb_id: str = ""
    doc_id: str = ""
    corpus_version: int = 0
    index_version: int = 0

    def to_dict(self) -> dict:
        return {
            "law": self.law,
            "chapter_path": self.chapter_path,
            "article_no": self.article_no,
            "score": round(float(self.score), 4),
            "snippet": self.snippet,
            "chunk_id": self.chunk_id,
            "kind": self.kind,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "version": self.version,
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
        }


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    # 来源 snippet 也可能带内部路径/密钥（脏文档），出前端前统一脱敏
    text = redact_sensitive(text)[0]
    return text[:limit] + ("……" if len(text) > limit else "")


def sources_from_qa(
    qa: QAPair | None,
    *,
    similarity: float | None = None,
    is_reference: bool = False,
) -> list[SourceRef]:
    """问答标准答案的来源：``law_refs`` 逐条展开为 SourceRef。

    ``is_reference=True`` 表示这是同类参考（未精确命中），相关度置低。
    """
    if qa is None:
        return []
    if is_reference:
        score = REFERENCE_SCORE
    else:
        score = similarity if similarity is not None else 1.0
    out: list[SourceRef] = []
    for ref in qa.law_refs or []:
        law = (ref.get("law") or "").strip()
        article = (ref.get("article") or "").strip()
        if not law and not article:
            continue
        out.append(
            SourceRef(
                law=law or "高频问答库",
                chapter_path="",
                article_no=article,
                score=score,
                snippet=_snippet(qa.answer),
                kind="qa",
                kb_id=qa.kb_id or "",
            )
        )
    if not out:
        # law_refs 缺失时至少给一个库级来源，前端不至于没有引用可展示
        out.append(
            SourceRef(
                law="高频问答库",
                chapter_path="",
                article_no="",
                score=score,
                snippet=_snippet(qa.answer),
                kind="qa",
                kb_id=qa.kb_id or "",
            )
        )
    return out


def sources_from_hits(hits: Sequence | None) -> list[SourceRef]:
    """检索命中（``app.retrieval.hybrid.Hit``）→ SourceRef。"""
    out: list[SourceRef] = []
    for h in hits or []:
        score = h.rerank_score if h.rerank_score is not None else h.score
        domain = profile_name_for_kb(getattr(h, "kb_id", "") or "")
        out.append(
            SourceRef(
                law=h.law_name,
                chapter_path=h.chapter_path,
                article_no="" if domain == "generic" else h.article_no,
                score=score,
                snippet=_snippet(h.parent_content or h.content),
                chunk_id=h.chunk_id,
                kind="law" if domain == "legal" else "section",
                effective_from=getattr(h, "effective_from", ""),
                effective_to=getattr(h, "effective_to", ""),
                version=getattr(h, "version", 0),
                kb_id=getattr(h, "kb_id", "") or "",
                doc_id=getattr(h, "doc_id", "") or "",
                corpus_version=getattr(h, "corpus_version", 0) or 0,
                index_version=getattr(h, "index_version", 0) or 0,
            )
        )
    return out


def aggregate_sources(
    *,
    qa_hit: QAPair | None = None,
    similarity: float | None = None,
    references: Sequence[QAPair] | None = None,
    hits: Sequence | None = None,
    limit: int = 10,
) -> list[SourceRef]:
    """合并问答 + 同类参考 + 检索来源：去重（法名#条号；无条号用块 ID）、按分数降序、限量。"""
    merged = (
        sources_from_qa(qa_hit, similarity=similarity)
        + [r for qa in (references or []) for r in sources_from_qa(qa, is_reference=True)]
        + sources_from_hits(hits)
    )
    best: dict[tuple[str, str], SourceRef] = {}
    for ref in merged:
        # 产品资料（generic）没有条号，同一手册的多个章节必须用块 ID 区分，
        # 否则会被当成同一条来源互相覆盖。
        key = (ref.law, ref.article_no or ref.chunk_id)
        if key not in best or ref.score > best[key].score:
            best[key] = ref
    ordered = sorted(best.values(), key=lambda r: r.score, reverse=True)
    return ordered[:limit] if limit > 0 else ordered


def build_context_sources(
    *,
    qa_hit: QAPair | None = None,
    similarity: float | None = None,
    references: Sequence[QAPair] | None = None,
    hits: Sequence | None = None,
) -> list[ContextSource]:
    """prompt 参考资料：QA 整段答案（精确命中 + 同类参考）+ 法条父块全文。

    与 :func:`aggregate_sources` 的区别：这里保留完整正文供 LLM 阅读；
    那里只保留 snippet 供前端展示。两者的去重与排序口径一致。
    """
    out: list[ContextSource] = []
    if qa_hit is not None:
        out.append(
            ContextSource(
                law="高频问答库",
                chapter_path="",
                article_no="",
                content=qa_hit.answer.strip(),
                score=similarity if similarity is not None else 1.0,
                kind="qa",
            )
        )
    for qa in references or []:
        out.append(
            ContextSource(
                law=f"高频问答库·{qa.category}" if qa.category else "高频问答库",
                chapter_path="",
                article_no="",
                content=f"问：{qa.question_std}\n答：{qa.answer.strip()}",
                score=REFERENCE_SCORE,
                kind="qa",
            )
        )
    for h in hits or []:
        domain = profile_name_for_kb(getattr(h, "kb_id", "") or "")
        out.append(
            ContextSource(
                law=h.law_name,
                chapter_path=h.chapter_path,
                article_no="" if domain == "generic" else h.article_no,
                content=(h.parent_content or h.content or "").strip(),
                score=h.rerank_score if h.rerank_score is not None else h.score,
                chunk_id=h.chunk_id,
                kind="law" if domain == "legal" else "section",
                domain=domain,
                effective_from=getattr(h, "effective_from", ""),
                effective_to=getattr(h, "effective_to", ""),
                version=getattr(h, "version", 0),
            )
        )
    return out


def qa_conclusion_block(qa: QAPair) -> str:
    """混合链路的"结论段"：结构化标准答案置前，法条作依据段（§6.3）。"""
    return (
        "【结论参考】以下为高频问答库中的标准答案，可作为结论段的基础，"
        "但表述需结合下方法条资料复核：\n"
        f"问题：{qa.question_std}\n答案：{qa.answer.strip()}"
    )
