from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kvresearch.navigator import Action
from kvresearch.reporting import write_stage_report


class StageReportingTests(unittest.TestCase):
    def test_writes_compact_markdown_with_cycle_and_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "research"
            run.mkdir(parents=True)
            action = Action(
                "READY",
                "monitor",
                "command",
                (str(run / "cycles" / "014" / "attempts" / "002" / "run_manifest.json"),),
                str(run / "cycles" / "014" / "attempts" / "002" / "exit.json"),
                ("kvresearch", "monitor", "--cycle", "014", "--attempt", "002"),
            )
            report = write_stage_report(
                root / "reports",
                run,
                action,
                {"status": "failed", "failures": ["terminated"], "large_payload": list(range(1000))},
            )
            text = report.read_text(encoding="utf-8")
            self.assertEqual(report.suffix, ".md")
            self.assertEqual(report.name, "kvresearch_stage_report.md")
            self.assertIn("cycle014 / 开发评测 attempt002", text)
            self.assertIn("技术失败", text)
            self.assertNotIn("large_payload", text)

            second = write_stage_report(
                root / "reports",
                run,
                Action("READY", "aggregate", "command", (), str(run / "cycles" / "014" / "metrics.json")),
                {"status": "complete"},
            )
            self.assertEqual(second, report)
            text = report.read_text(encoding="utf-8")
            self.assertEqual(len(list((root / "reports").glob("*.md"))), 1)
            self.assertIn("cycle014 / 开发评测 attempt002", text)
            self.assertIn("cycle014 / 结果聚合", text)
            self.assertIn("## 当前阶段", text)
            self.assertIn("## 时间线", text)
            self.assertIn("- 当前状态：cycle014 / 结果聚合", text)
            self.assertEqual(text.count("|---|---|---|---|"), 1)

    def test_plan_report_includes_compact_locked_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "runs" / "research"
            cycle = run / "cycles" / "015"
            cycle.mkdir(parents=True)
            plan = cycle / "experiment_plan.json"
            plan.write_text(json.dumps({
                "evaluation_tier": "formal",
                "datasets": ["qasper", "gsm8k"],
                "budgets": {"gpu_count": 8, "batch_size_schedule": [16, 8, 4, 2, 1]},
                "decoding": {"math_max_total_length": 2048},
                "storage_accounting": {"theoretical_average_bits": 1.0},
            }), encoding="utf-8")
            report = write_stage_report(
                root / "reports",
                run,
                Action("READY", "experiment_plan", "agent", (), str(plan)),
                {"status": "complete"},
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("`tier=formal`", text)
            self.assertIn("`tasks=2`", text)
            self.assertIn("`gpu_count=8`", text)


if __name__ == "__main__":
    unittest.main()
