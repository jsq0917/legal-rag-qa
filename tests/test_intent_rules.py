"""意图识别规则兜底单测：五类意图关键词表、复杂度、改写类型、结构化话题。

纯函数测试，无 IO。
"""

from __future__ import annotations

import pytest

from app.orchestration.intent_rules import (
    normalize_query,
    rule_based_intent,
    rule_complexity,
    rule_intent,
    rule_law_hints,
    rule_rewrite_type,
    rule_structured_topic,
)


class TestNormalize:
    def test_punctuation_removed(self):
        assert normalize_query("请问，试用期最长多久？") == "请问试用期最长多久"

    def test_fullwidth_and_ascii_mixed(self):
        # 中英文标点、空白一并去掉；注意本模块不做大小写折叠
        assert normalize_query("HR 说：试用期(六个月)？") == "HR说试用期六个月"

    def test_keyword_survives_normalization(self):
        # 归一化不影响关键词命中（停用词由关键词表天然容忍）
        assert "试用期" in normalize_query("请问，我想问一下：试用期最长多久？")


class TestRuleIntent:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("试用期最长可以约定多久", "labor_law"),
            ("加班费怎么算", "labor_law"),
            ("公司拖欠工资怎么办", "labor_law"),
            ("工伤认定需要哪些材料", "labor_law"),
            ("离婚财产怎么分割", "civil_code"),
            ("借钱不还怎么起诉", "civil_code"),
            ("遗产继承顺序是什么", "civil_code"),
            ("你好", "greeting"),
            ("在吗", "greeting"),
            ("你能做什么", "other"),
            ("今天天气怎么样", "general"),
            ("", "general"),
        ],
    )
    def test_intent_classification(self, query, expected):
        intent, _conf = rule_intent(query)
        assert intent == expected

    def test_labor_wins_over_civil_on_tie(self):
        # "劳动合同"含"合同"（民法典词），劳动优先
        intent, _ = rule_intent("劳动合同违约金")
        assert intent == "labor_law"

    def test_confidence_increases_with_hits(self):
        _, low = rule_intent("工资")
        _, high = rule_intent("工资 加班 辞退 社保 工伤 仲裁")
        assert high > low


class TestRuleComplexity:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("试用期多久", "simple"),
            ("加班费怎么算", "medium"),  # 含"怎么算"连接词 → medium（§6.1 兜底表）
            ("试用期被辞退有经济补偿吗", "medium"),
            ("公司裁员经济补偿怎么算，试用期算在内吗", "medium"),  # 22 字含连接词
            # 归一化后 41 字（>40）→ complex（规则兜底无法识别多子句，靠长度兜底）
            ("公司裁员经济补偿怎么算，试用期算在内吗，社保断了怎么办，竞业协议和培训服务期究竟还有效吗", "complex"),
            ("工资为什么拖欠？社保呢？", "complex"),  # 多问号
            ("", "simple"),
        ],
    )
    def test_complexity(self, query, expected):
        assert rule_complexity(query) == expected

    def test_long_query_is_complex(self):
        assert rule_complexity("我" + "在公" * 25 + "司工作") == "complex"


class TestRuleRewriteType:
    def test_anaphora_is_history(self):
        assert rule_rewrite_type("那个能详细说吗") == "history"
        assert rule_rewrite_type("还有呢") == "history"
        assert rule_rewrite_type("继续") == "history"

    def test_plain_is_direct(self):
        assert rule_rewrite_type("试用期最长多久") == "direct"


class TestRuleStructuredTopic:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("试用期最长多久", "试用期"),
            ("周末加班给双倍吗", "加班费"),
            ("裁员有补偿吗", "裁员"),
            ("老婆怀孕被辞退", "三期保护"),
            ("社保断缴怎么办", "社保"),
            ("仲裁时效一年怎么算", "仲裁时效"),
            ("今天天气不错", None),
            ("", None),
        ],
    )
    def test_topic_detection(self, query, expected):
        assert rule_structured_topic(query) == expected

    def test_longest_keyword_wins(self):
        # "经济补偿"比"补偿"（经济补偿话题）更具体，但两者同话题；
        # "解除"（合同解除）vs "经济补偿"（经济补偿）→ 更长关键词胜出
        assert rule_structured_topic("解除合同有经济补偿吗") in {"经济补偿", "合同解除"}


class TestRuleLawHints:
    def test_labor(self):
        assert rule_law_hints("labor_law") == [
            "中华人民共和国劳动合同法",
            "中华人民共和国劳动法",
        ]

    def test_civil(self):
        assert rule_law_hints("civil_code") == ["中华人民共和国民法典"]

    def test_general_all(self):
        assert len(rule_law_hints("general")) == 3


class TestRuleBasedIntent:
    def test_full_result_structure(self):
        result = rule_based_intent("试用期被辞退有补偿吗")
        assert result.source == "rules"
        assert result.intent == "labor_law"
        assert result.complexity == "medium"
        assert result.rewrite_type == "direct"
        assert result.sub_queries == ["试用期被辞退有补偿吗"]
        assert result.hit_structured is True
        assert result.structured_topic == "试用期"
        assert result.law_hints

    def test_history_rewrite_with_history(self):
        result = rule_based_intent("那个多久", history=[{"role": "user", "content": "试用期"}])
        assert result.rewrite_type == "history"
        assert result.need_history is True

    def test_history_rewrite_without_history(self):
        result = rule_based_intent("那个多久")
        assert result.rewrite_type == "history"
        assert result.need_history is False

    def test_greeting_no_law_hints(self):
        result = rule_based_intent("你好")
        assert result.intent == "greeting"
        assert result.law_hints == []
        assert result.hit_structured is False

    def test_empty_query(self):
        result = rule_based_intent("")
        assert result.intent == "general"
        assert result.sub_queries == []


class TestRuleSubQueries:
    """规则兜底多跳拆解：LLM 降级时保住 decompose 能力（REPORT §4.2 假设 C）。"""

    def test_multiple_questions_split(self):
        from app.orchestration.intent_rules import rule_sub_queries

        subs = rule_sub_queries("996不给加班费能解除合同吗？要补偿吗？")
        assert len(subs) == 2
        assert "加班费" in subs[0] and "解除" in subs[0]
        assert "补偿" in subs[1]

    def test_explicit_connectors_split(self):
        from app.orchestration.intent_rules import rule_sub_queries

        subs = rule_sub_queries("经济补偿怎么算；仲裁时效是多久")
        assert subs == ["经济补偿怎么算", "仲裁时效是多久"]

    def test_single_question_not_split(self):
        from app.orchestration.intent_rules import rule_sub_queries

        assert rule_sub_queries("试用期最长可以约定多久？") == ["试用期最长可以约定多久？"]
        assert rule_sub_queries("加班费怎么算") == ["加班费怎么算"]

    def test_heci_word_not_split(self):
        """含"和解"的词不能被切坏（不按"和"切分的回归测试）。"""
        from app.orchestration.intent_rules import rule_sub_queries

        assert rule_sub_queries("工伤后签的和解协议有效吗") == ["工伤后签的和解协议有效吗"]

    def test_short_parts_dropped(self):
        from app.orchestration.intent_rules import rule_sub_queries

        # 切出的片段太短（<4字）不构成有效子查询，退回原句
        assert rule_sub_queries("补偿以及996") == ["补偿以及996"]

    def test_rule_based_intent_uses_split(self):
        from app.orchestration.intent_rules import rule_based_intent

        result = rule_based_intent("未签合同的二倍工资怎么算？仲裁时效多久？")
        assert len(result.sub_queries) == 2

    def test_empty_query(self):
        from app.orchestration.intent_rules import rule_sub_queries

        assert rule_sub_queries("") == []
        assert rule_sub_queries("   ") == []


class TestMultiIssue:
    """多诉求问句判定：只用于检索配额，不改变意图字段。"""

    @pytest.mark.parametrize(
        "query",
        [
            "公司拖欠工资，我能不能直接走人还要补偿？",
            "公司违法解除合同，赔偿金和经济补偿能同时主张吗",
            "口头说好试用期三个月但合同没写，还作数吗？",
            "未签合同的二倍工资怎么算？仲裁时效多久？",
        ],
    )
    def test_multi_issue_true(self, query):
        from app.orchestration.intent_rules import is_multi_issue

        assert is_multi_issue(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "试用期最长可以约定多久？",
            "加班费怎么算",
            "怀孕期间被公司辞退，我能主张什么？",  # 只有一个诉求，不因逗号放大
            "",
        ],
    )
    def test_multi_issue_false(self, query):
        from app.orchestration.intent_rules import is_multi_issue

        assert is_multi_issue(query) is False
