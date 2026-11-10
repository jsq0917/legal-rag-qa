"""服务端授权范围与缓存版本标识。

为什么单独成模块：

* 缓存（答案 / 问答精确 / 意图）必须在 key 上携带"谁能看、看到哪一版语料"，
  否则同一句话换知识库、换租户、换索引版本就会互相回放；
* 检索、父块回表、来源聚合、调试端点都需要同一份"最终可见 kb 集合"，
  不能各算各的；
* 授权范围必须由服务端从数据库解析，客户端传入的 ``kb_ids`` 只能缩小范围。

本模块只做**纯计算**与一次性授权解析，不依赖 FastAPI，方便离线单测。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from sqlalchemy import text

#: 缓存 key 结构版本。任何字段语义变化都必须 +1，让旧缓存自然不可达。
CACHE_SCHEMA_VERSION = 2

#: 默认租户（种子知识库与历史用户都落在这里）
DEFAULT_TENANT = "default"

ROLE_ANONYMOUS = "anonymous"
ROLE_VIEWER = "viewer"
ROLE_EDITOR = "editor"
ROLE_ADMIN = "admin"

#: 角色能力等级（只读 < 编辑 < 管理员）
ROLE_RANK = {
    ROLE_ANONYMOUS: 0,
    ROLE_VIEWER: 1,
    ROLE_EDITOR: 2,
    ROLE_ADMIN: 3,
}


@dataclass(frozen=True)
class KbGrant:
    """服务端解析出的一个知识库授权快照。"""

    kb_id: str
    corpus_version: int = 1
    index_version: int = 1
    is_public: bool = True
    tenant_id: str = DEFAULT_TENANT
    owner_id: int | None = None


@dataclass(frozen=True)
class AccessScope:
    """一次请求的最终可见范围。

    ``legacy=True`` 表示调用方没有提供服务端范围（仅测试 / 内部脚本使用），
    此时不做任何 kb 限制，但缓存 key 仍带 ``legacy`` 版本，不会与线上
    带范围的结果混用。
    """

    grants: tuple[KbGrant, ...] = ()
    tenant_id: str = DEFAULT_TENANT
    user_id: int | None = None
    role: str = ROLE_ANONYMOUS
    requested_kb_ids: tuple[str, ...] = ()
    denied_kb_ids: tuple[str, ...] = ()
    legacy: bool = False

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def legacy_all(cls) -> "AccessScope":
        """测试 / 内部调用：不做范围限制。"""
        return cls(legacy=True)

    @classmethod
    def for_grants(
        cls,
        grants: Iterable[KbGrant],
        *,
        tenant_id: str = DEFAULT_TENANT,
        user_id: int | None = None,
        role: str = ROLE_ADMIN,
        requested_kb_ids: Sequence[str] = (),
        denied_kb_ids: Sequence[str] = (),
    ) -> "AccessScope":
        return cls(
            grants=tuple(sorted(grants, key=lambda g: g.kb_id)),
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            requested_kb_ids=tuple(requested_kb_ids),
            denied_kb_ids=tuple(denied_kb_ids),
        )

    # ------------------------------------------------------------------
    # 范围运算
    # ------------------------------------------------------------------
    @property
    def kb_ids(self) -> tuple[str, ...]:
        return tuple(g.kb_id for g in self.grants)

    @property
    def has_grants(self) -> bool:
        return not self.legacy

    def restrict(self, kb_ids: Sequence[str] | None) -> list[str]:
        """把任意 kb 列表收敛到本范围；``None``/空 表示"范围内全部"。

        这是"客户端 kb_ids 只能缩小范围"的唯一实现点：任何越权 id 都会被
        丢弃，绝不会因为客户端多传一个 id 就扩大检索范围。
        """
        if self.legacy:
            return list(kb_ids or [])
        allowed = set(self.kb_ids)
        if not kb_ids:
            return list(self.kb_ids)
        return [kb for kb in kb_ids if kb in allowed]

    def allows_kb(self, kb_id: str | None) -> bool:
        if not kb_id:
            return False
        if self.legacy:
            return True
        return kb_id in self.kb_ids

    def grant(self, kb_id: str) -> KbGrant | None:
        for g in self.grants:
            if g.kb_id == kb_id:
                return g
        return None

    def can_read(self, kb_id: str | None) -> bool:
        return self.allows_kb(kb_id)

    def can_write(self, kb_id: str | None) -> bool:
        """写权限：管理员 / 同租户编辑；只读用户与匿名一律拒绝。"""
        if self.legacy:
            return True
        grant = self.grant(kb_id or "")
        if grant is None:
            return False
        if self.role == ROLE_ADMIN:
            return grant.tenant_id == self.tenant_id or grant.is_public
        if self.role == ROLE_EDITOR:
            return grant.tenant_id == self.tenant_id
        return False

    # ------------------------------------------------------------------
    # 缓存标识
    # ------------------------------------------------------------------
    def cache_namespace(self) -> str:
        if self.legacy:
            return "legacy"
        if self.grants and all(g.is_public for g in self.grants):
            # 全部是公共库：跨租户共享是安全的，答案内容一致
            return "public"
        return f"tenant:{self.tenant_id}"

    def cache_token(self) -> str:
        """范围 + 语料/索引版本的稳定指纹（不含问题本身）。"""
        if self.legacy:
            return "legacy"
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "namespace": self.cache_namespace(),
            "kbs": [
                [g.kb_id, int(g.corpus_version), int(g.index_version)]
                for g in self.grants
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        """调试用的范围描述（只有 id / 版本，不含标题、正文等敏感内容）。"""
        return {
            "tenant": self.tenant_id,
            "role": self.role,
            "namespace": self.cache_namespace(),
            "kb_ids": list(self.kb_ids),
            "versions": {g.kb_id: [g.corpus_version, g.index_version] for g in self.grants},
            "denied_count": len(self.denied_kb_ids),
        }

    def with_grants(self, grants: Iterable[KbGrant]) -> "AccessScope":
        return replace(self, grants=tuple(sorted(grants, key=lambda g: g.kb_id)))


def resolve_access_scope(
    db: Any,
    user: Any | None,
    requested_kb_ids: Sequence[str] | None = None,
) -> AccessScope:
    """从数据库解析当前用户的可见知识库。

    说明：本函数处于请求热路径，只发一条 ``SELECT``；解析失败时**不**降级为
    "全部可见"（那会把越权数据放出去），而是返回空范围，由调用方走无数据分支。
    """
    requested = tuple(dict.fromkeys([k for k in (requested_kb_ids or []) if k]))
    user_id = getattr(user, "id", None)
    tenant_id = getattr(user, "tenant_id", DEFAULT_TENANT) or DEFAULT_TENANT
    role = getattr(user, "role", ROLE_VIEWER) or (ROLE_VIEWER if user_id else ROLE_ANONYMOUS)
    if user is None:
        role = ROLE_ANONYMOUS

    try:
        rows = db.execute(
            text(
                "SELECT id, corpus_version, index_version, is_public, tenant_id, owner_id "
                "FROM knowledge_bases ORDER BY id"
            )
        ).mappings().all()
    except Exception:
        # 旧库或测试替身没有新列时，退回最小列集；此时全部按公共库处理，
        # 等迁移执行后会走上面的完整查询。
        try:
            rows = db.execute(
                text("SELECT id, corpus_version, index_version FROM knowledge_bases ORDER BY id")
            ).mappings().all()
        except Exception:
            rows = []

    grants: list[KbGrant] = []
    for row in rows:
        data = dict(row)
        is_public = bool(data.get("is_public", 1))
        kb_tenant = data.get("tenant_id") or DEFAULT_TENANT
        # 匿名用户只能看公共库；私库必须同租户（且已登录）。
        if not is_public and (user is None or kb_tenant != tenant_id):
            continue
        grants.append(
            KbGrant(
                kb_id=str(data["id"]),
                corpus_version=int(data.get("corpus_version") or 1),
                index_version=int(data.get("index_version") or 1),
                is_public=is_public,
                tenant_id=kb_tenant,
                owner_id=data.get("owner_id"),
            )
        )

    allowed_ids = {g.kb_id for g in grants}
    denied = [kb for kb in requested if kb not in allowed_ids]
    effective = [g for g in grants if not requested or g.kb_id in requested]
    return AccessScope.for_grants(
        effective,
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        requested_kb_ids=requested,
        denied_kb_ids=denied,
    )


def answer_identity(
    scope: AccessScope,
    *,
    model: str = "",
    prompt_version: str = "",
    as_of: str = "",
) -> str:
    """答案缓存完整身份：范围 + 模型 + prompt 版本 + 生效日期。"""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "scope": scope.cache_token(),
        "namespace": scope.cache_namespace(),
        "model": model or "",
        "prompt": prompt_version or "",
        "as_of": as_of or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


__all__ = [
    "AccessScope",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_TENANT",
    "KbGrant",
    "ROLE_ADMIN",
    "ROLE_ANONYMOUS",
    "ROLE_EDITOR",
    "ROLE_VIEWER",
    "answer_identity",
    "resolve_access_scope",
]
