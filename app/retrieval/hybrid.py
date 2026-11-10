"""混合检索：bge-m3 稠密 + 稀疏双路召回，RRF 融合。

为什么必须双路
--------------
法律问答里有大量"术语型"查询（代通知金、竞业限制、经济补偿、N+1），
稠密向量擅长语义近似但会把近义术语拉平；bge-m3 的稀疏权重本质是学习出来的
词项权重，能把这些术语精确顶上来。两路各取 4×topK 候选，用 RRF 融合——
RRF 只看排名不看分数，天然规避了余弦相似度与 IP 分数不可比的问题。

本模块只负责"召回+融合"，重排与父块回填在 rerank.py / service.py。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingest.embedder import Embedding
from app.ingest.versioning import int_to_date

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"

#: 合法知识库 id（与 POST /api/kbs 的创建规则一致）。kb_ids 会拼进 Milvus
#: 过滤表达式，带引号/特殊字符的值必须拒之门外，否则构成表达式注入面。
_KB_ID_RE = re.compile(r"^[a-z0-9_]+$")

#: 检索时返回的标量字段（避免把大字段全拉回来）
OUTPUT_FIELDS = [
    "chunk_id",
    "parent_id",
    "kb_id",
    "doc_id",
    "content",
    "title",
    "article_no",
    "chapter_path",
    "law_name",
    "source_file",
    "effective_from",
    "effective_to",
    "version",
]

#: 查询时 HNSW 的搜索宽度（比默认 16 高，保证 2k 规模下召回接近 1）
DENSE_EF = 64
#: 稀疏检索丢弃低权重维度的比例
SPARSE_DROP_RATIO = 0.2


@dataclass
class Hit:
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
    score: float = 0.0
    rerank_score: float | None = None
    parent_content: str = ""
    #: 时效元信息（YYYY-MM-DD，空=未知）与文档入库版本
    effective_from: str = ""
    effective_to: str = ""
    version: int = 0
    #: 知识库级版本快照（检索时补全，便于 trace/评估追溯）
    corpus_version: int = 0
    index_version: int = 0

    @property
    def embed_text(self) -> str:
        """重排输入：标题 + 正文，标题带法律名与章节。"""
        return f"{self.title}\n{self.content}" if self.title else self.content

    @property
    def law_article(self) -> str:
        """评估口径：`法名#条号`，跨法同号不会混淆。"""
        return f"{self.law_name}#{self.article_no}"


@dataclass
class RetrievalResult:
    hits: list[Hit]
    question: str
    topk: int
    pool: int
    complexity: str
    mode: str
    reranked: bool
    elapsed_ms: int
    debug: dict = field(default_factory=dict)


def build_kb_filter(kb_ids: list[str] | None) -> str:
    if not kb_ids:
        return ""
    # 只放行 [a-z0-9_]+：这类字符拼不进引号逃逸，注入面被格式校验关死
    valid = [kb for kb in kb_ids if _KB_ID_RE.match(kb or "")]
    if not valid:
        return ""
    quoted = ", ".join(f'"{kb}"' for kb in valid)
    return f"kb_id in [{quoted}]"


def build_filter(
    kb_ids: list[str] | None = None,
    law_hints: list[str] | None = None,
    *,
    effective_as_of: int = 0,
) -> str:
    """拼接 Milvus 过滤表达式：知识库 + 法域。

    法域过滤是本轮针对"两部劳动法术语高度重叠"加的：稀疏路按词项召回时，
    很容易把《劳动法》第二十一条顶到《劳动合同法》第十九条前面——
    问题问的是哪部法，意图层已经给出 ``law_hints``，直接用它把检索范围收窄。

    注意 ``law_hints`` 为空（general/other 意图）时不过滤，避免误伤召回。
    """
    parts: list[str] = []
    kb_expr = build_kb_filter(kb_ids)
    if kb_expr:
        parts.append(kb_expr)
    if law_hints:
        quoted = ", ".join(f'"{law}"' for law in law_hints)
        parts.append(f"law_name in [{quoted}]")
    if effective_as_of > 0:
        # 0 表示"未知/长期有效"：老数据没有日期时不过度过滤，避免整库消失。
        as_of = int(effective_as_of)
        parts.append(f"(effective_from == 0 or effective_from <= {as_of})")
        parts.append(f"(effective_to == 0 or effective_to >= {as_of})")
    return " and ".join(parts)


def to_hit(raw) -> Hit:
    """Milvus 命中 → Hit。兼容 MilvusClient 的 dict / Hit 两种返回形态。"""
    entity = raw.get("entity", raw) if isinstance(raw, dict) else getattr(raw, "entity", {})
    distance = raw.get("distance", 0.0) if isinstance(raw, dict) else getattr(raw, "distance", 0.0)
    return Hit(
        chunk_id=entity.get("chunk_id", ""),
        parent_id=entity.get("parent_id", ""),
        kb_id=entity.get("kb_id", ""),
        doc_id=entity.get("doc_id", ""),
        content=entity.get("content", ""),
        title=entity.get("title", ""),
        article_no=entity.get("article_no", ""),
        chapter_path=entity.get("chapter_path", ""),
        law_name=entity.get("law_name", ""),
        source_file=entity.get("source_file", ""),
        score=float(distance or 0.0),
        effective_from=int_to_date(entity.get("effective_from")),
        effective_to=int_to_date(entity.get("effective_to")),
        version=int(entity.get("version") or 0),
    )


def build_requests(
    embedding: Embedding,
    *,
    pool: int,
    expr: str,
    mode: str,
    sparse_pool_ratio: float = 1.0,
):
    """构造 Milvus 混合检索请求。mode=dense 时只发稠密一路（消融用）。

    ``sparse_pool_ratio`` < 1 时稀疏路少取候选（如 0.5 → 稀疏取 pool/2）：
    两路各取 pool 会让稀疏路挤占候选池名额，正确条连候选都进不了
    （REPORT §4.4 证伪加权 RRF 后的真实怀疑点，供消融验证）。
    """
    from pymilvus import AnnSearchRequest

    # 空稀疏向量会被 Milvus 拒绝；这类极短 query 降级为纯稠密
    indices, values = embedding.sparse_arrays()
    requests = [
        AnnSearchRequest(
            data=[embedding.dense],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": DENSE_EF}},
            limit=pool,
            expr=expr or None,
        )
    ]
    if mode == "hybrid" and indices:
        sparse_limit = pool if sparse_pool_ratio >= 1.0 else max(4, int(pool * sparse_pool_ratio))
        requests.append(
            AnnSearchRequest(
                data=[dict(zip(indices, values))],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "IP", "params": {"drop_ratio_search": SPARSE_DROP_RATIO}},
                limit=sparse_limit,
                expr=expr or None,
            )
        )
    return requests


def search(
    client,
    collection: str,
    embedding: Embedding,
    *,
    pool: int,
    kb_ids: list[str] | None = None,
    mode: str = "hybrid",
    rrf_k: int = 60,
    ranker: str = "weighted",
    dense_weight: float = 0.75,
    sparse_weight: float = 0.25,
    law_hints: list[str] | None = None,
    sparse_pool_ratio: float = 1.0,
    effective_as_of: int = 0,
) -> list[Hit]:
    """执行检索并融合。返回按融合分降序的候选（未重排）。

    融合策略由 ``ranker`` 决定（生产默认经 config 传 ``rrf`` 等权 RRF）：
    - ``rrf``：等权 RRF。消融结论（REPORT §4.4）：加权无收益已回退等权。
    - ``weighted``：Milvus ``WeightedRanker`` 按归一化分数加权（保留作对照）。
    """
    from pymilvus import RRFRanker, WeightedRanker

    expr = build_filter(kb_ids, law_hints, effective_as_of=effective_as_of)
    requests = build_requests(
        embedding, pool=pool, expr=expr, mode=mode, sparse_pool_ratio=sparse_pool_ratio
    )
    if len(requests) == 1:
        results = client.search(
            collection_name=collection,
            data=[embedding.dense],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": DENSE_EF}},
            limit=pool,
            filter=expr or None,
            output_fields=OUTPUT_FIELDS,
        )
    else:
        if ranker == "rrf":
            fusion = RRFRanker(k=rrf_k)
        else:
            fusion = WeightedRanker(dense_weight, sparse_weight)
        results = client.hybrid_search(
            collection_name=collection,
            reqs=requests,
            ranker=fusion,
            limit=pool,
            output_fields=OUTPUT_FIELDS,
        )
    if not results:
        return []
    return [to_hit(raw) for raw in results[0]]


def dedupe_by_parent(hits: list[Hit], *, topk: int) -> list[Hit]:
    """同一法条命中多个子块时只保留分最高的一条，其余内容合并进父块。"""
    best: dict[str, Hit] = {}
    order: list[str] = []
    for hit in hits:
        key = hit.parent_id or hit.chunk_id
        if key not in best:
            best[key] = hit
            order.append(key)
        elif _rank_value(hit) > _rank_value(best[key]):
            best[key] = hit
    # 按分数排序，保证无论输入是 RRF 序还是已重排序，输出都是"最优在前"
    ordered = sorted(order, key=lambda k: _rank_value(best[k]), reverse=True)
    return [best[key] for key in ordered][:topk]


def _rank_value(hit: Hit) -> float:
    """比较用分数：重排后看 rerank_score，否则看融合分。"""
    return hit.rerank_score if hit.rerank_score is not None else hit.score
