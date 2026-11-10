"""基础设施文件测试：docker-compose 端口规划、SQL 结构、初始化脚本切分。

这类"配置即代码"的东西最容易在后期悄悄改坏（比如改回 19530 撞端口），
用测试钉住比写文档更可靠。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from config import BASE_DIR

COMPOSE = BASE_DIR / "docker-compose.yml"
SCHEMA = BASE_DIR / "mysql" / "init" / "01_schema.sql"
SEED = BASE_DIR / "mysql" / "init" / "02_seed_kb.sql"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema_sql() -> str:
    return SCHEMA.read_text(encoding="utf-8")


class TestCompose:
    def test_project_name(self, compose):
        assert compose["name"] == "lawanswer"

    def test_no_legacy_version_key(self, compose):
        """compose v2 已废弃 version 字段，留着会有 warning。"""
        assert "version" not in compose

    def test_services(self, compose):
        assert set(compose["services"]) == {"mysql", "redis", "etcd", "minio", "milvus"}

    @pytest.mark.parametrize(
        "service,expected",
        [
            ("mysql", "3307:3306"),
            ("redis", "6380:6379"),
            ("milvus", "19531:19530"),
            ("minio", "9002:9000"),
        ],
    )
    def test_offset_ports(self, compose, service, expected):
        """端口必须避开本机 shopmate 占用的 3306/6379/19530/9000。"""
        assert expected in compose["services"][service]["ports"]

    def test_milvus_port_9092(self, compose):
        assert "9092:9091" in compose["services"]["milvus"]["ports"]

    def test_every_service_has_healthcheck(self, compose):
        for name, svc in compose["services"].items():
            assert "healthcheck" in svc, f"{name} 缺少 healthcheck"

    def test_milvus_depends_on_healthy_deps(self, compose):
        deps = compose["services"]["milvus"]["depends_on"]
        assert deps["etcd"]["condition"] == "service_healthy"
        assert deps["minio"]["condition"] == "service_healthy"

    def test_milvus_image_pinned(self, compose):
        assert compose["services"]["milvus"]["image"] == "milvusdb/milvus:v2.4.4"

    def test_mysql_init_mounted(self, compose):
        volumes = compose["services"]["mysql"]["volumes"]
        assert any("docker-entrypoint-initdb.d" in v for v in volumes)

    def test_passwords_from_env_not_hardcoded(self, compose):
        env = compose["services"]["mysql"]["environment"]
        assert env["MYSQL_ROOT_PASSWORD"].startswith("${MYSQL_ROOT_PASSWORD")
        assert env["MYSQL_PASSWORD"].startswith("${MYSQL_PASSWORD")

    def test_volumes_are_local_bind_mounts(self, compose):
        """数据落在 ./volumes，便于备份与彻底清理。"""
        for name in ("mysql", "redis", "milvus", "etcd", "minio"):
            volumes = compose["services"][name]["volumes"]
            assert any(v.startswith("./volumes/") for v in volumes), f"{name} 缺少数据卷"
            assert all(v.startswith("./") for v in volumes), f"{name} 存在非本地挂载"


class TestSchemaSql:
    def test_all_tables_defined(self, schema_sql):
        tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", schema_sql))
        assert tables == {
            "users",
            "knowledge_bases",
            "chat_sessions",
            "chat_messages",
            "qa_pairs",
            "documents",
            "document_chunks",
            "ingest_jobs",
        }

    def test_charset_utf8mb4(self, schema_sql):
        assert schema_sql.count("DEFAULT CHARSET = utf8mb4") == 8

    def test_users_unique_username(self, schema_sql):
        assert "UNIQUE KEY uk_username (username)" in schema_sql

    def test_sessions_index_supports_list_query(self, schema_sql):
        assert "idx_user_updated (user_id, deleted_at, updated_at DESC)" in schema_sql

    def test_history_rounds_default_five(self, schema_sql):
        assert "history_rounds TINYINT       NOT NULL DEFAULT 5" in schema_sql

    def test_qa_fulltext_uses_ngram(self, schema_sql):
        """中文全文检索必须 ngram，否则按空格分词根本匹配不到。"""
        assert re.search(r"FULLTEXT KEY ft_search \(search_text\) WITH PARSER ngram", schema_sql)

    def test_qa_pairs_have_search_text_not_json(self, schema_sql):
        """FULLTEXT 不支持 JSON 列，必须有冗余的 search_text 列。"""
        qa_block = schema_sql.split("CREATE TABLE IF NOT EXISTS qa_pairs")[1].split("ENGINE =")[0]
        assert "search_text       TEXT         NOT NULL" in qa_block

    def test_documents_incremental_unique_key(self, schema_sql):
        assert "UNIQUE KEY uk_kb_content (kb_id, content_sha1)" in schema_sql

    def test_documents_status_enum_covers_superseded(self, schema_sql):
        assert "'superseded'" in schema_sql

    def test_chunks_foreign_key_cascade(self, schema_sql):
        assert (
            "CONSTRAINT fk_chunk_doc FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE"
            in schema_sql
        )

    def test_chunk_trace_fields(self, schema_sql):
        for field in ("article_no", "chapter_path", "parent_id", "chunk_index"):
            assert field in schema_sql

    def test_messages_observability_columns(self, schema_sql):
        for column in ("intent", "route", "ttft_ms", "cached", "token_usage"):
            assert column in schema_sql

    def test_seed_creates_two_knowledge_bases(self):
        seed = SEED.read_text(encoding="utf-8")
        assert "'kb_labor'" in seed
        assert "'kb_civil'" in seed
        assert "INSERT IGNORE" in seed


class TestInitDb:
    def test_split_statements_drops_comments(self):
        from app.db.init_db import split_statements

        sql = "-- 注释\nCREATE TABLE a (id INT);\n-- 又一注释\nINSERT INTO a VALUES (1);\n"
        stmts = split_statements(sql)
        assert len(stmts) == 2
        assert stmts[0].startswith("CREATE TABLE")
        assert stmts[1].startswith("INSERT INTO")
        assert all("注释" not in s for s in stmts)

    def test_schema_file_splits_into_expected_statements(self, schema_sql):
        from app.db.init_db import split_statements

        stmts = split_statements(schema_sql)
        assert len(stmts) >= 9  # SET + 8 张表
        allowed = ("SET ", "CREATE ", "INSERT ", "ALTER ")
        assert all(s.strip().upper().startswith(allowed) for s in stmts)

    def test_rejects_unsafe_database_name(self, settings_factory, monkeypatch):
        """库名来自配置，必须防注入。"""
        import app.db.init_db as init_db

        monkeypatch.setattr(
            init_db, "get_settings", lambda: settings_factory(mysql_db="bad-name; DROP DATABASE x")
        )
        with pytest.raises(SystemExit):
            init_db.ensure_database()

    def test_sql_dir_exists(self):
        init_dir = BASE_DIR / "mysql" / "init"
        assert init_dir.is_dir()
        sqls = sorted(p.name for p in init_dir.glob("*.sql"))
        # 建表文件必须存在且按序第一个执行；允许后续阶段追加种子文件
        assert "01_schema.sql" in sqls
        assert "02_seed_kb.sql" in sqls
        assert all(s.endswith(".sql") for s in sqls)
        assert sqls[0] == "01_schema.sql"


class TestProjectLayout:
    def test_required_packages_exist(self):
        for pkg in ("app", "tests", "scripts", "mysql", "config.py", "pyproject.toml"):
            assert (BASE_DIR / pkg).exists(), f"缺少 {pkg}"

    def test_gitignore_excludes_env_and_logs(self):
        content = (BASE_DIR / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in content
        assert "logs/" in content
        assert "volumes/" in content
        assert "!.env.example" in content

    def test_env_example_has_no_real_secret(self):
        from config import assert_secrets_not_in_dotenv

        assert_secrets_not_in_dotenv(BASE_DIR / ".env.example")

    def test_data_sources_present(self):
        raw = {p.name for p in (BASE_DIR / "data" / "raw").glob("*.md")}
        assert raw == {
            "中华人民共和国劳动合同法.md",
            "中华人民共和国劳动法_中华人民共和国人力资源和社会保障部.md",
            "中华人民共和国民法典.md",
            "中华人民共和国劳动合同法实施条例.md",
            "中华人民共和国劳动争议调解仲裁法.md",
            "工伤保险条例.md",
            "职工带薪年休假条例.md",
        }
