"""LA-N1：多轮会话与缓存正确性评测工具（离线）。

锁住三件事：
1. 数据集版本化且 dev/holdout 分离；
2. fixture 口径产出逐轮结果、汇总指标与失败会话，语义判断不计入；
3. ``legacy`` → ``scoped`` 两种缓存身份在同一数据集上可复现
   "同问题不同知识库/租户错误回放" 的修复前后对照。
"""

from __future__ import annotations

import hashlib
import json

from app.eval import multiturn_eval as mt


def _sessions():
    return mt.load_sessions(mt.DEFAULT_DATASET)


class TestDataset:
    def test_versioned_and_split(self):
        sessions = _sessions()
        assert len(sessions) == 10
        assert {s["dataset_version"] for s in sessions} == {"2026-11-11.1"}
        dev = [s for s in sessions if s["split"] == "dev"]
        hold = [s for s in sessions if s["split"] == "holdout"]
        assert sum(len(s["turns"]) for s in dev) == 11
        assert sum(len(s["turns"]) for s in hold) == 4
        # 每轮都带期望改写与是否拒答，评测才有可比口径
        for session in sessions:
            for turn in session["turns"]:
                assert turn["expected_standalone"]
                assert isinstance(turn["should_refuse"], bool)
                assert "expected_refs" in turn


class TestEvaluation:
    def test_scoped_has_no_cache_false_hit_and_reports_failures(self):
        report = mt.evaluate(_sessions(), identity_mode="scoped")

        assert report["meta"]["mode"] == "fixture"
        assert report["meta"]["data_kind"] == "synthetic_fixture"
        assert report["meta"]["turns"] == 15
        cache = report["summary"]["cache_false_hit"]
        assert cache["probes"] == 4 and cache["false_hits"] == 0
        assert cache["rate"] == 0.0
        # 语义判断单独放置，不参与通过率
        assert report["summary"]["semantic_judge"]["enabled"] is False
        assert report["summary"]["semantic_judge"]["judged"] == 0
        # 失败会话能定位到轮次与原因
        assert "mt_dev_rewrite" in report["failed_sessions"]
        reasons = report["failed_sessions"]["mt_dev_rewrite"][0]["failures"]
        assert any("rewrite_mismatch" in reason for reason in reasons)
        assert len(report["raw_turns"]) == 15

    def test_legacy_reproduces_cross_kb_and_tenant_false_hits(self):
        report = mt.evaluate(_sessions(), identity_mode="legacy")
        cache = report["summary"]["cache_false_hit"]
        # 4 个探针中，跨库与跨租户各错回放 1 次
        assert cache["false_hits"] == 2
        assert cache["rate"] == 0.5
        failed = report["failed_sessions"]
        assert "mt_dev_cache_civil" in failed
        assert "mt_hold_cache_acme" in failed
        assert any("cache_false_hit" in r for r in failed["mt_dev_cache_civil"][0]["failures"])

    def test_before_after_comparison_is_reproducible(self, tmp_path, capsys):
        legacy_out = tmp_path / "legacy.json"
        assert mt.main(["--identity", "legacy", "--out", str(legacy_out)]) == 0
        scoped_out = tmp_path / "scoped.json"
        assert mt.main(["--identity", "scoped", "--out", str(scoped_out), "--compare", str(legacy_out)]) == 0

        legacy = json.loads(legacy_out.read_text(encoding="utf-8"))
        scoped = json.loads(scoped_out.read_text(encoding="utf-8"))
        assert legacy["meta"]["dataset_sha256"] == hashlib.sha256(
            mt.DEFAULT_DATASET.read_bytes()
        ).hexdigest()
        assert legacy["summary"]["cache_false_hit"]["rate"] == 0.5
        assert scoped["summary"]["cache_false_hit"]["rate"] == 0.0
        out = capsys.readouterr().out
        assert "cache_false_hit" in out
        assert "0.500 ->  0.000" in out

    def test_split_filter_only_counts_one_split(self):
        sessions = [s for s in _sessions() if s["split"] == "holdout"]
        report = mt.evaluate(sessions, identity_mode="scoped")
        assert report["meta"]["turns"] == 4
        assert report["meta"]["sessions"] == 3

    def test_live_mode_requires_key_and_never_fakes_metrics(self, capsys, monkeypatch):
        for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert mt.main(["--mode", "live"]) == 3
        err = capsys.readouterr().err
        assert "ERROR" in err and "LLM_API_KEY" in err
