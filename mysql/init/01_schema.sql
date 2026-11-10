-- ============================================================================
-- LawAnswer 数据库结构  v1.0
--
-- 执行方式（二选一）：
--   1) 容器首次初始化：docker compose 会把本目录挂到 /docker-entrypoint-initdb.d
--      由 MySQL 官方镜像自动执行（自动选中 MYSQL_DATABASE=lawanswer）
--   2) 手动/重复执行：uv run python -m app.db.init_db（幂等，全部 IF NOT EXISTS）
--
-- 约定：
--   * 所有表 utf8mb4 / utf8mb4_0900_ai_ci（中文排序 + emoji 安全）
--   * 时间统一 DATETIME(3)，毫秒精度，避免"同一秒内排序不稳定"
--   * 会话用软删除（deleted_at），消息/块用硬删除 + 外键级联
-- ============================================================================

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- 1. 用户
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
    username      VARCHAR(64)     NOT NULL                COMMENT '账号名（3-32 位字母数字下划线）',
    password_hash VARCHAR(100)    NOT NULL                COMMENT 'bcrypt 哈希（自带盐，不单独存 salt）',
    status        TINYINT         NOT NULL DEFAULT 1      COMMENT '1 正常 0 禁用',
    last_login_at DATETIME(3)     NULL                     COMMENT '最近登录时间',
    created_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at    DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_username (username)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户';

-- ---------------------------------------------------------------------------
-- 2. 知识库（id 直接作为 Milvus partition key）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id              VARCHAR(32)  NOT NULL                COMMENT '语义化 slug，如 kb_labor',
    name            VARCHAR(64)  NOT NULL                COMMENT '展示名，如 劳动法知识库',
    description     VARCHAR(255) NULL,
    embedding_model VARCHAR(64)  NOT NULL DEFAULT 'bge-m3',
    doc_count       INT          NOT NULL DEFAULT 0      COMMENT '冗余计数，列表页免 count',
    chunk_count     INT          NOT NULL DEFAULT 0,
    created_by      BIGINT UNSIGNED NULL,
    created_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at      DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_name (name)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '知识库';

-- ---------------------------------------------------------------------------
-- 3. 对话会话（参数与会话绑定，切会话自动回填）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    id             CHAR(26)      NOT NULL                COMMENT 'ULID，按时间有序',
    user_id        BIGINT UNSIGNED NOT NULL,
    title          VARCHAR(128)  NOT NULL DEFAULT '新对话',
    kb_ids         JSON          NOT NULL                COMMENT '勾选的知识库 id 数组',
    model          VARCHAR(64)   NOT NULL DEFAULT 'deepseek-v4.1-flash',
    temperature    DECIMAL(3,2)  NOT NULL DEFAULT 0.30,
    top_p          DECIMAL(3,2)  NOT NULL DEFAULT 0.80,
    max_tokens     INT           NOT NULL DEFAULT 2048,
    history_rounds TINYINT       NOT NULL DEFAULT 5      COMMENT '历史对话轮数，默认 5',
    message_count  INT           NOT NULL DEFAULT 0,
    deleted_at     DATETIME(3)   NULL                     COMMENT '软删除',
    created_at     DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at     DATETIME(3)   NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    KEY idx_user_updated (user_id, deleted_at, updated_at DESC),
    CONSTRAINT fk_session_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '对话会话';

-- ---------------------------------------------------------------------------
-- 4. 消息（同时是评估数据源：intent/route/ttft_ms/cached 全落库）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_messages (
    id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    session_id   CHAR(26)        NOT NULL,
    seq          INT             NOT NULL                COMMENT '会话内序号，从 1 递增',
    role         ENUM('user','assistant','system') NOT NULL,
    content      MEDIUMTEXT      NOT NULL,
    sources      JSON            NULL                     COMMENT '法条来源列表',
    intent       VARCHAR(32)     NULL,
    rewrite_type VARCHAR(16)     NULL,
    route        VARCHAR(16)     NULL                     COMMENT 'structured/vector/hybrid/fallback',
    cached       TINYINT         NOT NULL DEFAULT 0,
    latency_ms   INT             NULL,
    ttft_ms      INT             NULL                     COMMENT '首 token 时间',
    token_usage  JSON            NULL,
    error        VARCHAR(255)    NULL                     COMMENT '流式中断等异常',
    created_at   DATETIME(3)     NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    KEY idx_session_seq (session_id, seq),
    KEY idx_created (created_at),
    CONSTRAINT fk_message_session FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '对话消息';

-- ---------------------------------------------------------------------------
-- 5. 高频问答对（结构化链路事实源）
--    注意：MySQL 的 FULLTEXT 索引不支持 JSON 列，因此额外冗余 search_text
--          （= 标准问法 + 所有口语化变体，换行拼接）供 ngram 全文检索。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qa_pairs (
    id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    category          VARCHAR(32)  NOT NULL              COMMENT '试用期/加班费/经济补偿/...',
    question_std      VARCHAR(255) NOT NULL              COMMENT '标准问法',
    question_variants JSON         NOT NULL              COMMENT '口语化变体数组',
    search_text       TEXT         NOT NULL              COMMENT '标准问法+变体拼接，供 FULLTEXT',
    answer            TEXT         NOT NULL              COMMENT '标准答案（含法条引用）',
    law_refs          JSON         NOT NULL              COMMENT '法条引用数组',
    kb_id             VARCHAR(32)  NULL,
    intent            VARCHAR(32)  NOT NULL DEFAULT 'labor_law',
    keywords          VARCHAR(255) NULL                  COMMENT '逗号分隔，规则兜底用',
    hit_count         INT          NOT NULL DEFAULT 0,
    is_active         TINYINT      NOT NULL DEFAULT 1,
    created_at        DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at        DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    KEY idx_category (category, is_active),
    KEY idx_hit (hit_count DESC),
    KEY idx_kb (kb_id, is_active),
    FULLTEXT KEY ft_search (search_text) WITH PARSER ngram
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '高频问答对';

-- ---------------------------------------------------------------------------
-- 6. 文档（增量更新的锚点：content_sha1 作版本号）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            CHAR(26)     NOT NULL                COMMENT 'ULID',
    kb_id         VARCHAR(32)  NOT NULL,
    filename      VARCHAR(255) NOT NULL,
    file_type     VARCHAR(16)  NOT NULL                COMMENT 'pdf/md/pptx/png/jpg',
    file_size     BIGINT       NOT NULL DEFAULT 0,
    content_sha1  CHAR(40)     NOT NULL                COMMENT '归一化正文 SHA1 —— 增量判定主键',
    file_sha1     CHAR(40)     NOT NULL                COMMENT '原始字节 SHA1 —— 防重复上传',
    storage_path  VARCHAR(512) NULL,
    status        ENUM('pending','parsing','chunking','embedding','ready','failed','superseded')
                  NOT NULL DEFAULT 'pending',
    progress      TINYINT      NOT NULL DEFAULT 0,
    error_msg     VARCHAR(512) NULL,
    parse_config  JSON         NOT NULL                COMMENT 'chunk_size/overlap/retrieval_mode',
    article_count INT          NOT NULL DEFAULT 0,
    chunk_count   INT          NOT NULL DEFAULT 0,
    version       INT          NOT NULL DEFAULT 1,
    effective_from DATE         NULL                    COMMENT '施行日期，空=未知',
    effective_to   DATE         NULL                    COMMENT '失效日期，空=长期有效',
    created_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at    DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_kb_content (kb_id, content_sha1),
    KEY idx_kb_status (kb_id, status),
    KEY idx_kb_file (kb_id, filename),
    CONSTRAINT fk_document_kb FOREIGN KEY (kb_id) REFERENCES knowledge_bases (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '文档';

-- ---------------------------------------------------------------------------
-- 7. 文档块（MySQL 为事实源，支撑预览/编辑/删除；Milvus 存同一份子块向量）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id          VARCHAR(64) NOT NULL                COMMENT 'doc_id:p{条号} / doc_id:p{条号}:c{i}',
    doc_id      CHAR(26)    NOT NULL,
    kb_id       VARCHAR(32) NOT NULL,
    parent_id   VARCHAR(64) NOT NULL                COMMENT '子块指向父块，父块自指',
    chunk_type  ENUM('parent','child') NOT NULL,
    chunk_index INT         NOT NULL,
    article_no  VARCHAR(32) NULL                     COMMENT '第十九条',
    chapter_path VARCHAR(255) NULL                   COMMENT '第一编·第二章·第一节',
    title       VARCHAR(255) NULL                    COMMENT '生成上下文用的标题行',
    content     TEXT        NOT NULL,
    char_count  INT         NOT NULL DEFAULT 0,
    effective_from DATE     NULL                     COMMENT '施行日期，空=未知',
    effective_to   DATE     NULL                     COMMENT '失效日期，空=长期有效',
    version     INT         NOT NULL DEFAULT 1       COMMENT '文档入库版本',
    in_milvus   TINYINT     NOT NULL DEFAULT 1       COMMENT '0 = 已编辑待重嵌入',
    is_enabled  TINYINT     NOT NULL DEFAULT 1       COMMENT '软删除/禁用',
    created_at  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    updated_at  DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    KEY idx_doc_index (doc_id, chunk_index),
    KEY idx_parent (parent_id),
    KEY idx_kb_article (kb_id, article_no),
    CONSTRAINT fk_chunk_doc FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '文档块';

-- ---------------------------------------------------------------------------
-- 8. 摄入任务（大文件异步进度，民法典 1260 条需要 3-8 分钟）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id         CHAR(26)    NOT NULL,
    doc_id     CHAR(26)    NOT NULL,
    stage      VARCHAR(32) NOT NULL                COMMENT 'parse/clean/chunk/embed/write',
    status     ENUM('running','success','failed') NOT NULL DEFAULT 'running',
    processed  INT         NOT NULL DEFAULT 0,
    total      INT         NOT NULL DEFAULT 0,
    message    VARCHAR(512) NULL,
    started_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    finished_at DATETIME(3) NULL,
    PRIMARY KEY (id),
    KEY idx_doc_started (doc_id, started_at DESC),
    CONSTRAINT fk_job_doc FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '摄入任务';
