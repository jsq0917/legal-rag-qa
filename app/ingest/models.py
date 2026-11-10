"""摄入层数据结构。

刻意保持"纯数据"：解析、清洗、分块各步之间只靠这些 dataclass 传递，
不夹带 ORM 对象或 Milvus 实体，方便单测与后续替换存储。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadedDocument:
    """加载器产物：一段纯文本 + 来源元信息。"""

    source_file: str
    text: str
    file_type: str
    page_count: int = 0
    #: 扫描件/图片等没有文本层的页数，用于提示"需要 OCR"
    empty_pages: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def is_ocr_needed(self) -> bool:
        return self.page_count > 0 and self.empty_pages == self.page_count


@dataclass(frozen=True)
class Article:
    """一条完整法条（父块的原型）。"""

    article_no: str          # 第十九条
    article_num: int         # 19（排序/校验用）
    chapter_path: str        # 第二编·第二章·第一节（缺级省略）
    content: str             # 条文正文（不含"第X条"头）
    line_start: int
    line_end: int

    def make_title(self, law_name: str) -> str:
        """生成带法律名的标题，防两部法律同条号张冠李戴。"""
        parts = [law_name] if law_name else []
        if self.chapter_path:
            parts.append(self.chapter_path)
        parts.append(self.article_no)
        return "·".join(parts)


@dataclass(frozen=True)
class ParsedDocument:
    """解析+结构化后的文档。"""

    source_file: str
    law_name: str
    text: str                # 清洗归一后的全文（用于算 content_sha1）
    articles: list[Article]
    skipped_articles: int = 0

    @property
    def article_count(self) -> int:
        return len(self.articles)


@dataclass(frozen=True)
class Chunk:
    """写入 MySQL / Milvus 的分块。"""

    chunk_id: str
    doc_id: str
    kb_id: str
    parent_id: str
    chunk_type: str          # parent / child
    chunk_index: int
    article_no: str
    chapter_path: str
    title: str
    content: str
    char_count: int
    #: 法条时效（YYYY-MM-DD，空串=未知）；version 为文档入库版本
    effective_from: str = ""
    effective_to: str = ""
    version: int = 1

    @property
    def embed_text(self) -> str:
        """向量化输入：标题 + 正文。

        标题里带法律名与章节，能显著提升"劳动合同法第三十九条"这类
        带法名查询的召回；正文保持纯净，便于前端编辑与人工校对。
        """
        return f"{self.title}\n{self.content}" if self.title else self.content
