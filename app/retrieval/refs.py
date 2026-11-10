"""法条显式互引：从条文里抽"依照本法第 X 条"这类引用，多跳时沿边补检。

只抽**显式**引用：必须带"本法/本条例/《法名》"或"依照/违反/符合…"等提示词。
法条正文里裸的"第 X 条"可能是正文叙述或目录残留，不当作边，避免把噪声
法条硬塞进候选；"第 X 款/第 X 项"也不是条级引用。

解析结果按 ``(法名, 条号数字)`` 索引到父块。索引全量加载（当前语料 1.5k 条、
约 0.1MB），进程内缓存 60 秒，新入库文档最迟 60 秒后可见；测试可显式清理。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

from sqlalchemy import text

from app.ingest.normalize import cn_to_int
from app.ingest.structure import law_name_from_filename
from app.retrieval.hybrid import Hit

logger = logging.getLogger(__name__)

_CN = "零一二三四五六七八九十百千"
#: 单条引用里的条号；用于解析列表/区间尾部的"、第四十一条"等
_ARTICLE_NUM_RE = re.compile(rf"第([{_CN}]+)条")
#: 显式引用：可选提示词 + 可选法名/本法 + 第X条 + 可选列表尾巴
_REF_RE = re.compile(
    rf"(?:(?P<cue>依照|依据|根据|按照|违反|符合|适用|参照)\s*)?"
    rf"(?:《(?P<law>[^》]{{2,40}})》\s*|(?P<local>本法|本条例)\s*)?"
    rf"第(?P<num>[{_CN}]+)条"
    rf"(?P<tail>(?:\s*[、和及至]\s*第[{_CN}]+条)*)"
)

#: 单条法条最多抽多少条边，防止"本法"密集引用把候选池灌满
MAX_REFS_PER_ARTICLE = 12
#: 索引缓存 TTL（秒）
CACHE_TTL = 60.0


@dataclass(frozen=True)
class ArticleRef:
    """一条显式引用。``law_name=None`` 表示"本法/本条例"，解析时沿用源条法名。"""

    law_name: str | None
    article_num: int
    cue: str


@dataclass(frozen=True)
class ParentArticle:
    """父块（整条法条）的路由索引行。"""

    chunk_id: str
    parent_id: str
    kb_id: str
    doc_id: str
    content: str
    title: str
    article_no: str
    chapter_path: str
    law_name: str
    source_file: str


def normalize_law_name(name: str) -> str:
    """法名归一：去书名号/空白，剥掉"中华人民共和国"前缀，便于别名匹配。"""
    name = (name or "").strip().strip("《》").replace(" ", "")
    prefix = "中华人民共和国"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name


def article_num_of(article_no: str) -> int:
    """``第四十二条`` → 42；``段1`` → 0（非标准条文，不参与互引）。"""
    match = _ARTICLE_NUM_RE.search(article_no or "")
    return cn_to_int(match.group(1)) if match else 0


def extract_refs(content: str, *, law_name: str, article_num: int) -> list[ArticleRef]:
    """抽取一条法条正文里的显式引用，按出现顺序去重。"""
    refs: list[ArticleRef] = []
    seen: set[tuple[str | None, int]] = set()
    for match in _REF_RE.finditer(content or ""):
        law = match.group("law")
        local = match.group("local")
        cue = match.group("cue")
        # 仅裸"第X条"、无任何引用信号的，视为正文叙述，不建边
        if not (law or local or cue):
            continue
        raw_nums = [match.group("num")]
        raw_nums += _ARTICLE_NUM_RE.findall(match.group("tail") or "")
        for raw in raw_nums:
            num = cn_to_int(raw)
            if num <= 0 or num == article_num:
                continue
            key = (normalize_law_name(law) if law else None, num)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ArticleRef(law_name=law, article_num=num, cue=cue or local or ""))
            if len(refs) >= MAX_REFS_PER_ARTICLE:
                return refs
    return refs


class ArticleIndex:
    """``(法名, 条号)`` → 父块。同法同号只保留第一条（重复导入的旧文档禁用后不会进来）。"""

    def __init__(self, rows: list[ParentArticle]):
        self.by_key: dict[tuple[str, int], ParentArticle] = {}
        self.law_names: dict[str, str] = {}
        for row in rows:
            num = article_num_of(row.article_no)
            if num <= 0:
                continue
            self.by_key.setdefault((row.law_name, num), row)
            self.law_names.setdefault(normalize_law_name(row.law_name), row.law_name)

    def resolve(self, ref: ArticleRef, source_law: str) -> ParentArticle | None:
        if ref.law_name:
            law = self.law_names.get(normalize_law_name(ref.law_name))
            if not law:
                return None
        else:
            law = source_law
        return self.by_key.get((law, ref.article_num))

    def __len__(self) -> int:
        return len(self.by_key)


_CACHE: dict[str, tuple[float, ArticleIndex]] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _load_rows(settings) -> list[ParentArticle]:
    from app.db.mysql import get_engine

    stmt = text(
        "SELECT c.id, c.parent_id, c.kb_id, c.doc_id, c.content, c.title, "
        "c.article_no, c.chapter_path, d.filename "
        "FROM document_chunks c JOIN documents d ON d.id = c.doc_id "
        "WHERE c.chunk_type = 'parent' AND c.is_enabled = 1"
    )
    with get_engine(settings).connect() as conn:
        rows = conn.execute(stmt).fetchall()
    out: list[ParentArticle] = []
    for row in rows:
        filename = row[8] or ""
        out.append(
            ParentArticle(
                chunk_id=row[0],
                parent_id=row[1],
                kb_id=row[2],
                doc_id=row[3],
                content=row[4] or "",
                title=row[5] or "",
                article_no=row[6] or "",
                chapter_path=row[7] or "",
                law_name=law_name_from_filename(filename) if filename else "",
                source_file=filename,
            )
        )
    return out


def get_article_index(settings=None, *, force: bool = False) -> ArticleIndex:
    """取进程内缓存的互引索引（TTL 60s，失败返回空索引而非打断检索）。"""
    from config import get_settings

    s = settings or get_settings()
    key = s.mysql.dsn
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and not force and now - cached[0] < CACHE_TTL:
            return cached[1]
    try:
        index = ArticleIndex(_load_rows(s))
    except Exception as exc:  # noqa: BLE001 - 互引索引是增强项，失败只降级
        logger.warning("法条互引索引加载失败，跳过扩展：%s", str(exc)[:200])
        return ArticleIndex([])
    with _CACHE_LOCK:
        _CACHE[key] = (now, index)
    return index


def _row_to_hit(row: ParentArticle, *, seed: Hit) -> Hit:
    """把被引父块转成 Hit：分数略低于来源条，保证插在源条之后。"""
    return Hit(
        chunk_id=row.chunk_id,
        parent_id=row.parent_id,
        kb_id=row.kb_id,
        doc_id=row.doc_id,
        content=row.content,
        title=row.title,
        article_no=row.article_no,
        chapter_path=row.chapter_path,
        law_name=row.law_name,
        source_file=row.source_file,
        score=seed.score * 0.99,
        rerank_score=None,
        parent_content=row.content,
    )


def expand_hits_with_refs(
    hits: list[Hit],
    *,
    topk: int,
    kb_ids: list[str] | None = None,
    max_sources: int = 3,
    max_added: int = 3,
    index: ArticleIndex | None = None,
    settings=None,
) -> tuple[list[Hit], int]:
    """沿头部命中的显式引用补检被引法条，插在来源条之后。

    只扩展前 ``max_sources`` 条来源、最多补 ``max_added`` 条；已命中的法条
    不重复加入。返回值第二项是实际新增条数（0 表示没有可用边）。
    """
    if not hits or max_added <= 0:
        return hits, 0
    idx = index if index is not None else get_article_index(settings)
    if not len(idx):
        return hits, 0

    used = {hit.parent_id or hit.chunk_id for hit in hits}
    allowed_kbs = set(kb_ids) if kb_ids else None
    out: list[Hit] = []
    added = 0
    for position, hit in enumerate(hits):
        out.append(hit)
        if position >= max_sources or added >= max_added:
            continue
        source_text = hit.parent_content or hit.content
        refs = extract_refs(
            source_text, law_name=hit.law_name, article_num=article_num_of(hit.article_no)
        )
        for ref in refs:
            if added >= max_added:
                break
            row = idx.resolve(ref, hit.law_name)
            if row is None or row.parent_id in used:
                continue
            if allowed_kbs is not None and row.kb_id not in allowed_kbs:
                continue
            out.append(_row_to_hit(row, seed=hit))
            used.add(row.parent_id)
            added += 1
    return out[:topk] if topk > 0 else out, added
