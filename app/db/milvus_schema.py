"""Milvus collection 定义与初始化。

隔离方案：**单 collection + ``kb_id`` 作 partition key**。
理由：跨库检索一次请求即可完成、schema 只维护一份、按库删除/重建高效；
相比"一库一 collection"少了 N 张表迁移的麻烦。

存储策略：Milvus **只存子块**（检索单元），父块正文留在 MySQL，
命中后按 ``parent_id`` 批量回表，避免整条法条在向量库重复存两份。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema

#: bge-m3 稠密向量维度
DENSE_DIM = 1024

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
PRIMARY_FIELD = "chunk_id"
PARTITION_KEY_FIELD = "kb_id"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    max_length: int | None = None
    dim: int | None = None
    is_primary: bool = False
    is_partition_key: bool = False
    description: str = ""


@dataclass(frozen=True)
class IndexSpec:
    field_name: str
    index_type: str
    metric_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


#: 字段定义（顺序即建表顺序）
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(PRIMARY_FIELD, "VARCHAR", max_length=64, is_primary=True, description="块 ID，与 MySQL document_chunks.id 一致"),
    FieldSpec(PARTITION_KEY_FIELD, "VARCHAR", max_length=32, is_partition_key=True, description="知识库 ID，分区键"),
    FieldSpec("doc_id", "VARCHAR", max_length=26, description="文档 ID，增量重建时按此删除"),
    FieldSpec("parent_id", "VARCHAR", max_length=64, description="父块 ID，用于回表取整条法条"),
    FieldSpec("content", "VARCHAR", max_length=4096, description="子块正文"),
    FieldSpec("title", "VARCHAR", max_length=255, description="法律名·章节·条号"),
    FieldSpec("law_name", "VARCHAR", max_length=64, description="法律名称"),
    FieldSpec("article_no", "VARCHAR", max_length=32, description="条号，如 第十九条"),
    FieldSpec("chapter_path", "VARCHAR", max_length=255, description="编/章/节路径"),
    FieldSpec("source_file", "VARCHAR", max_length=255, description="溯源文件名"),
    FieldSpec("chunk_index", "INT32", description="文档内序号"),
    FieldSpec("char_count", "INT32", description="字符数"),
    FieldSpec("doc_sha1", "VARCHAR", max_length=40, description="内容版本号"),
    FieldSpec("updated_at", "INT64", description="毫秒时间戳"),
    FieldSpec("effective_from", "INT64", description="施行日期 YYYYMMDD，0=未知"),
    FieldSpec("effective_to", "INT64", description="失效日期 YYYYMMDD，0=长期有效"),
    FieldSpec("version", "INT32", description="文档入库版本"),
    FieldSpec(DENSE_FIELD, "FLOAT_VECTOR", dim=DENSE_DIM, description="bge-m3 稠密向量"),
    FieldSpec(SPARSE_FIELD, "SPARSE_FLOAT_VECTOR", description="bge-m3 稀疏向量（learned lexical weights）"),
)

#: 索引定义。注意：partition key 字段（kb_id）本身不支持建索引，故不在此列出。
INDEX_SPECS: tuple[IndexSpec, ...] = (
    IndexSpec(DENSE_FIELD, "HNSW", "COSINE", {"M": 16, "efConstruction": 200}),
    IndexSpec(SPARSE_FIELD, "SPARSE_INVERTED_INDEX", "IP", {"drop_ratio_search": 0.2}),
    IndexSpec("doc_id", "INVERTED"),
    IndexSpec("article_no", "INVERTED"),
    IndexSpec("source_file", "INVERTED"),
    IndexSpec("law_name", "INVERTED"),
)

_DTYPE_MAP: dict[str, DataType] = {
    "VARCHAR": DataType.VARCHAR,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
}


def to_field_schema(spec: FieldSpec) -> FieldSchema:
    kwargs: dict[str, Any] = {
        "name": spec.name,
        "dtype": _DTYPE_MAP[spec.dtype],
        "is_primary": spec.is_primary,
        "description": spec.description,
    }
    if spec.max_length is not None:
        kwargs["max_length"] = spec.max_length
    if spec.dim is not None:
        kwargs["dim"] = spec.dim
    if spec.is_partition_key:
        kwargs["is_partition_key"] = True
    return FieldSchema(**kwargs)


def build_schema(dim: int = DENSE_DIM) -> CollectionSchema:
    """构造 collection schema；``dim`` 允许覆盖以便将来换 embedding 模型。"""
    fields: list[FieldSchema] = []
    for spec in FIELD_SPECS:
        if spec.dim is not None and spec.dtype == "FLOAT_VECTOR":
            spec = FieldSpec(
                name=spec.name,
                dtype=spec.dtype,
                max_length=spec.max_length,
                dim=dim,
                is_primary=spec.is_primary,
                is_partition_key=spec.is_partition_key,
                description=spec.description,
            )
        fields.append(to_field_schema(spec))
    return CollectionSchema(
        fields=fields,
        description="LawAnswer 法律条文块（稠密+稀疏双向量）",
        enable_dynamic_field=False,
    )


def describe_specs() -> dict[str, Any]:
    """给测试/文档用的纯字典视图，不依赖 Milvus 连接。"""
    return {
        "dense_dim": DENSE_DIM,
        "fields": [
            {
                "name": s.name,
                "dtype": s.dtype,
                "max_length": s.max_length,
                "dim": s.dim or (DENSE_DIM if s.dtype == "FLOAT_VECTOR" else None),
                "is_primary": s.is_primary,
                "is_partition_key": s.is_partition_key,
            }
            for s in FIELD_SPECS
        ],
        "indexes": [
            {
                "field": i.field_name,
                "index_type": i.index_type,
                "metric_type": i.metric_type,
                "params": i.params,
            }
            for i in INDEX_SPECS
        ],
    }


def ensure_database(client, db_name: str) -> bool:
    """创建 Milvus database（存在则跳过），返回是否新建。"""
    if not db_name or db_name == "default":
        return False
    existing = client.list_databases()
    if db_name in existing:
        return False
    client.create_database(db_name)
    return True


def ensure_collection(
    client,
    db_name: str,
    collection_name: str,
    *,
    dim: int = DENSE_DIM,
    drop: bool = False,
) -> dict[str, Any]:
    """幂等创建 database + collection + 索引，并 load 到内存。"""
    created_db = ensure_database(client, db_name)
    if db_name and db_name != "default":
        client.using_database(db_name)

    existed = client.has_collection(collection_name)
    if existed and drop:
        client.drop_collection(collection_name)
        existed = False
    if existed:
        client.load_collection(collection_name)
        return {"created": False, "database": db_name, "collection": collection_name, "dropped": drop}

    index_params = client.prepare_index_params()
    for spec in INDEX_SPECS:
        kwargs: dict[str, Any] = {"field_name": spec.field_name, "index_type": spec.index_type}
        if spec.metric_type:
            kwargs["metric_type"] = spec.metric_type
        if spec.params:
            kwargs["params"] = dict(spec.params)
        index_params.add_index(**kwargs)

    client.create_collection(
        collection_name=collection_name,
        schema=build_schema(dim=dim),
        index_params=index_params,
        consistency_level="Strong",
    )
    client.load_collection(collection_name)
    return {
        "created": True,
        "database": db_name,
        "collection": collection_name,
        "created_database": created_db,
    }
