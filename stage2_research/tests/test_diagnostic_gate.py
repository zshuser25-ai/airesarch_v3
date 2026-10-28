from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kvresearch.diagnostics import diagnostic_contract, validate_diagnostic_outputs
from kvresearch.initialize import initialize_run
from kvresearch.navigator import Navigator
from kvresearch.pipeline import aggregate, decide, diagnose, learn
from kvresearch.storage import vecinfer_reference_bytes
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref, read_jsonl
from tests.helpers import test_config, write_control


class DiagnosticGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = test_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _plan(self, cycle: Path, run_id: str, tasks: list[str]):
        return write_control(
            cycle / "experiment_plan.json", "experiment_plan", run_id,
            candidate_id="diagnostic-candidate", code_scope=["x.py"],
            evaluation_tier="development", datasets=tasks, decoding={"math_max_total_length": 2048}, budgets={},
            decision_rules={
                "development_quality": "This diagnostic is non-promotable and cannot open formal data."
            },
            diagnostic_tests={"reproduction": "compare observer values"},
            outputs=[{"kind": "paired_arms_diagnostics_storage_and_joins"}],
            storage_accounting={"theoretical_average_bits": 1.0},
            workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"},
        )

    def _raw(self, attempt: Path, task: str):
        reference = vecinfer_reference_bytes(self.config, 256, 16)
        rows = [
            {
                "sample_idx": index, "dataset": task, "prompt": f"p{index}",
                "prediction": "a", "answers": ["a"], "score": 50.0,
                "prefill_tokens": 240, "output_tokens": 16, "total_tokens": 256,
                "actual_pq_kv_cache_bytes": reference,
            }
            for index in range(32)
        ]
        candidate = attempt / "raw" / f"{task}.jsonl"
        control = attempt / "arms" / task / "unmodified_vecinfer.jsonl"
        atomic_write_jsonl(candidate, rows)
        atomic_write_jsonl(control, rows)
        return candidate, control

    def _diagnostic(self, attempt: Path, task: str, candidate: Path, control: Path):
        path = attempt / "diagnostics" / f"{task}.json"
        atomic_write_json(path, {
            "task": task,
            "complete": True,
            "input_binding": {
                "candidate": path_ref(candidate),
                "control": path_ref(control),
            },
            "tests": {"reproduction": {"complete": True}},
            "checkpoints": [
                {"status": "observed", "layer": 0, "qk_delta": 0.5},
                {"status": "not_applicable", "layer": 16, "reason_code": 1},
            ],
            "comparisons": [
                {"metric": "qk_logit_l2", "candidate": 0.75, "control": 0.25, "delta": 0.5}
            ],
        })
        return path

    def test_prose_non_promotable_is_binding(self):
        contract = diagnostic_contract({
            "decision_rules": {"quality": "This diagnostic cannot itself be promoted."}
        })
        self.assertTrue(contract["required"])
        self.assertTrue(contract["non_promotable"])

    def test_missing_observer_evidence_invalidates_measurement(self):
        run = initialize_run(self.config, "diagnostic gate", "diagnostic-missing")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        self._plan(cycle, run.name, ["qasper"])
        write_control(cycle / "validation.json", "validation", run.name, all_hard_checks_pass=True)
        write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=0,
                      ended_at="x", elapsed_seconds=1, timed_out=False, oom=False,
                      completed_tasks=["qasper"], missing_tasks=[], log_hashes={})
        self._raw(attempt, "qasper")

        metrics = aggregate(self.config, run, "000")
        self.assertEqual(metrics["status"], "failed")
        self.assertEqual(metrics["error_code"], "E_DIAGNOSTICS_INCOMPLETE")
        self.assertFalse(metrics["diagnostic_evidence"]["valid"])
        write_control(cycle / "verified_registry.json", "verified_registry", run.name,
                      values=[], rejected_claims=[], source_hashes={})
        diagnosis = diagnose(self.config, run, "000")
        self.assertFalse(diagnosis["measurement_valid"])
        self.assertIn("diagnostic_evidence_incomplete",
                      {item["type"] for item in diagnosis["deficiencies"]})
        write_control(cycle / "selection.json", "selection", run.name,
                      selected_candidate_id="diagnostic-candidate",
                      ranking=["diagnostic-candidate", "c2", "c3", "c4"],
                      rejected=[], evidence=[])
        implementation = write_control(
            cycle / "implementation_manifest.json", "implementation_manifest", run.name,
            workspace=str(cycle / "workspace"), base_hash="a", changed_files=[],
            patch_hash="b", build_command=[], run_entrypoint="x", dependencies=[],
        )
        action = Navigator(self.config)._post_execution(
            run, cycle, cycle / "experiment_plan.json",
            cycle / "implementation_manifest.json", cycle / "validation.json",
            attempt / "exit.json",
        )
        self.assertEqual(action.step, "execution_repair")
        self.assertFalse((cycle / "decision.json").exists())

    def test_complete_hash_bound_finite_comparison_passes(self):
        run = initialize_run(self.config, "diagnostic gate", "diagnostic-valid")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        plan = self._plan(cycle, run.name, ["qasper"])
        candidate, control = self._raw(attempt, "qasper")
        evidence = self._diagnostic(attempt, "qasper", candidate, control)

        result = validate_diagnostic_outputs(plan, attempt)

        self.assertTrue(result["valid"])
        self.assertEqual(result["tasks"]["qasper"]["comparison_count"], 1)
        self.assertIn(str(evidence), result["evidence_paths"])

    def test_remaining_diagnostics_use_separate_namespace_and_hash_gate(self):
        run = initialize_run(self.config, "remaining diagnostic gate", "diagnostic-remaining")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        plan = self._plan(cycle, run.name, ["qasper"])
        candidate = attempt / "raw_remaining" / "qasper.jsonl"
        control = attempt / "arms_remaining" / "qasper" / "unmodified_vecinfer.jsonl"
        atomic_write_jsonl(candidate, [{"sample_idx": 0}])
        atomic_write_jsonl(control, [{"sample_idx": 0}])
        diagnostic = attempt / "diagnostics_remaining" / "qasper.json"
        atomic_write_json(diagnostic, {
            "task": "qasper", "complete": True,
            "input_binding": {"candidate": path_ref(candidate), "control": path_ref(control)},
            "tests": {"reproduction": {"complete": True}},
            "comparisons": [{"candidate": 1.0, "control": 0.0}],
        })

        result = validate_diagnostic_outputs(plan, attempt, remaining=True)
        self.assertTrue(result["valid"])
        self.assertEqual(result["phase"], "remaining_70pct")
        candidate.write_text('{"sample_idx": 1}\n', encoding="utf-8")
        drifted = validate_diagnostic_outputs(plan, attempt, remaining=True)
        self.assertFalse(drifted["valid"])
        self.assertTrue(any("hash drift" in error for error in drifted["errors"]))

    def test_selector_requires_real_coverage_and_independent_oracle(self):
        run = initialize_run(self.config, "selector gate", "selector-gate")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        plan = self._plan(cycle, run.name, ["qasper"])
        plan["diagnostic_tests"] = {
            "selector_rule": "independent FP32 reference",
        }
        candidate, control = self._raw(attempt, "qasper")
        diagnostic = attempt / "diagnostics" / "qasper.json"
        common = {
            "task": "qasper",
            "complete": True,
            "input_binding": {
                "candidate": path_ref(candidate),
                "control": path_ref(control),
            },
            "tests": {"selector_rule": {"complete": True}},
            "comparisons": [
                {"metric": "selector_score", "candidate": 0.5, "control": 0.6}
            ],
            "applicable_rollovers": 8,
            "changed_rollovers": 2,
            # A bare assertion must not satisfy the oracle contract.
            "selector_reference_oracle_agreement": True,
        }
        atomic_write_json(diagnostic, common)
        invalid = validate_diagnostic_outputs(plan, attempt)
        self.assertFalse(invalid["valid"])
        self.assertFalse(invalid["tasks"]["qasper"]["selector_semantics"]["oracle_valid"])

        atomic_write_json(diagnostic, {
            **common,
            "selector_reference_oracle": {
                "independent_fp32": True,
                "checks": 8,
                "mismatches": 0,
                "agreement": True,
            },
        })
        valid = validate_diagnostic_outputs(plan, attempt)
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["tasks"]["qasper"]["selector_semantics"]["oracle_checks"], 8)

    def test_valid_non_promotable_diagnostic_forces_pivot_and_lesson(self):
        run = initialize_run(self.config, "diagnostic decision", "diagnostic-pivot")
        cycle = run / "cycles" / "000"
        self._plan(cycle, run.name, ["qasper"])
        diagnostic = {
            "required": True, "non_promotable": True, "valid": True,
            "evidence_paths": [str(cycle / "attempts" / "000" / "diagnostics" / "qasper.json")],
            "tasks": {"qasper": {"valid": True, "comparison_count": 1}},
        }
        write_control(
            cycle / "metrics.json", "metrics", run.name,
            tasks={"qasper": {"score": 50.0}}, longbench={"score": 50.0},
            gsm8k={"score": 10.0}, math500={"score": 10.0}, math={"score": 10.0},
            theoretical_storage_pass=True, actual_storage_pass=True, failures=[],
            comparison_mode="development_candidate_ranking",
            baseline_reference={"longbench": {"qasper": 40.0},
                                "math": {"gsm8k": 0.0, "math500": 2.0}},
            diagnostic_evidence=diagnostic,
        )
        write_control(cycle / "verified_registry.json", "verified_registry", run.name,
                      values=[], rejected_claims=[], source_hashes={})
        write_control(
            cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name,
            deficiencies=[], severity="info", evidence=diagnostic["evidence_paths"],
            repairable=False, suggested_actions=[], measurement_valid=True,
            diagnostic_evidence=diagnostic,
        )

        decision = decide(self.config, run, "000")
        self.assertEqual(decision["action"], "PIVOT")
        self.assertIn("explore a different quality mechanism", decision["rationale"])
        self.assertFalse(decision["cooldown"]["small_gain"])
        self.assertEqual(decision["diagnostic_evidence"]["paths"], diagnostic["evidence_paths"])
        rows = learn(run, "000")
        diagnostic_lessons = [row for row in rows if row["lesson_id"].endswith("-diagnostic")]
        self.assertEqual(len(diagnostic_lessons), 1)
        self.assertEqual(diagnostic_lessons[0]["evidence"], diagnostic["evidence_paths"])


if __name__ == "__main__":
    unittest.main()
