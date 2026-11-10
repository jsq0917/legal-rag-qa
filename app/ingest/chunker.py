"""父子分块：父块按"第X条"边界，子块按条内语义单元。

为什么不是普通固定窗口
----------------------
本机语料实测：1,465 条法条平均 **97 字**、中位 76 字、P90 186 字、最长 454 字。
若子块也按 500 字切，子块数（≈1466）几乎等于父块数（1465），父子分块直接退化成 1:1；
而固定窗口 500/50 只能切出约 324 块，必然把"第四十七条"这类条文劈成两半。

所以策略是：

* **父块 = 一条完整法条**，永不被截断（保证进 prompt 的条文是完整的）；
* **子块 = 条内语义单元**，长度超过 `chunk_size` 才切，按句号/分号/项号断句，
  相邻子块带 `overlap` 重叠；短条文不强行切，子块内容等于父块；
* 检索命中子块后按 `parent_id` 回表取整条，既保精度又保完整。
"""

from __future__ import annotations

import re
from dataclasses import replace

from app.ingest.models import Article, Chunk, ParsedDocument

SENTENCE_END = "。；！？!?;"
#: 项/款编号：（一）1. ① 等，切分时优先在此断开
ITEM_RE = re.compile(r"(?=(?:（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\)|^\d+[.、]))")


def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句，标点保留在前一句。"""
    sentences: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_END:
            sentences.append("".join(buf))
            buf = []
    if buf:
        sentences.append("".join(buf))
    return [s for s in sentences if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """单句超长时按长度硬切（保底，避免子块无限长）。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def pack_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把句子贪心装箱成不超过 chunk_size 的片段，相邻片段保留句子级重叠。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    flattened: list[str] = []
    for sentence in sentences:
        if len(sentence) > chunk_size:
            flattened.extend(_hard_split(sentence, chunk_size))
        else:
            flattened.append(sentence)

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in flattened:
        if current and current_len + len(sentence) > chunk_size:
            pieces.append("".join(current))
            # 重叠按"整句回带"：把上一块末尾若干句带进下一块，
            # 保证重叠内容不会从半句话开始，也不会把新块撑爆。
            carry: list[str] = []
            carried_len = 0
            for prev in reversed(current):
                if carried_len + len(prev) <= overlap:
                    carry.insert(0, prev)
                    carried_len += len(prev)
                else:
                    break
            while carry and carried_len + len(sentence) > chunk_size:
                carried_len -= len(carry.pop(0))
            current = carry + [sentence]
            current_len = carried_len + len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        pieces.append("".join(current))
    return pieces


def split_content(content: str, chunk_size: int, overlap: int) -> list[str]:
    """把一条条文切成 1..N 个子块内容。"""
    if len(content) <= chunk_size:
        return [content]
    return pack_sentences(split_sentences(content), chunk_size, overlap)


def _plain_articles(
    doc: ParsedDocument, chunk_size: int, *, fake_labels: bool = True
) -> list[Article]:
    """非法律文书没有条号：按段落聚合。

    ``fake_labels=True`` 保持历史行为（段N 占位，法律库里的 PPT 等退化文档）；
    ``fake_labels=False`` 用于产品资料库，不伪造任何条号/段号。
    """
    paragraphs = [p.strip() for p in doc.text.splitlines() if p.strip()]
    blocks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > chunk_size * 2:
            blocks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}".strip()
    if current:
        blocks.append(current)
    return [
        Article(
            article_no=f"段{i + 1}" if fake_labels else "",
            article_num=i + 1,
            chapter_path="",
            content=block,
            line_start=0,
            line_end=0,
        )
        for i, block in enumerate(blocks)
    ]


def _section_articles(doc: ParsedDocument, chunk_size: int) -> list[Article]:
    """产品 FAQ/手册：优先按 Markdown 标题分节，章节即来源定位。"""
    lines = doc.text.splitlines()
    if not any(re.match(r"^#{1,6}\s+\S", line) for line in lines):
        return _plain_articles(doc, chunk_size, fake_labels=False)

    sections: list[Article] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            sections.append(
                Article(
                    article_no="",
                    article_num=len(sections) + 1,
                    chapter_path=heading,
                    content=content,
                    line_start=0,
                    line_end=0,
                )
            )

    for line in lines:
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            flush()
            heading = match.group(1).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return sections


def build_chunks(
    doc: ParsedDocument,
    *,
    doc_id: str,
    kb_id: str,
    chunk_size: int = 300,
    overlap: int = 50,
    include_parent: bool = True,
    label_mode: str = "article",
) -> list[Chunk]:
    """生成父块 + 子块。子块是检索单元，父块是生成单元。

    ``label_mode="section"`` 用于产品 FAQ/手册：按标题分节，不写条号；
    默认 ``article`` 保持法条父块行为不变。
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap({overlap}) 必须小于 chunk_size({chunk_size})")

    if doc.articles:
        articles = doc.articles
    elif label_mode == "section":
        articles = _section_articles(doc, chunk_size)
    else:
        articles = _plain_articles(doc, chunk_size)
    chunks: list[Chunk] = []
    index = 0
    for article in articles:
        parent_id = f"{doc_id}:p{article.article_num}"
        title = article.make_title(doc.law_name)
        if include_parent:
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc_id,
                    kb_id=kb_id,
                    parent_id=parent_id,
                    chunk_type="parent",
                    chunk_index=index,
                    article_no=article.article_no,
                    chapter_path=article.chapter_path,
                    title=title,
                    content=article.content,
                    char_count=len(article.content),
                )
            )
            index += 1

        for i, piece in enumerate(split_content(article.content, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{parent_id}:c{i}",
                    doc_id=doc_id,
                    kb_id=kb_id,
                    parent_id=parent_id,
                    chunk_type="child",
                    chunk_index=index,
                    article_no=article.article_no,
                    chapter_path=article.chapter_path,
                    title=title,
                    content=piece,
                    char_count=len(piece),
                )
            )
            index += 1
    return chunks


def truncation_rate(chunks: list[Chunk]) -> float:
    """子块覆盖不全整条法条的比例（评估指标：法条截断率）。"""
    parents = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}
    children: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.chunk_type == "child":
            children.setdefault(chunk.parent_id, []).append(chunk)
    if not parents:
        return 0.0
    truncated = 0
    for parent_id, parent in parents.items():
        kids = children.get(parent_id, [])
        merged = "".join(sorted(k.content for k in kids))
        if len(merged) < parent.char_count * 0.95:
            truncated += 1
    return truncated / len(parents)


def with_doc(doc: ParsedDocument, law_name: str) -> ParsedDocument:
    """替换法名（上传时知识库可能要求覆盖自动识别的法名）。"""
    return replace(doc, law_name=law_name)
