from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kvresearch.initialize import initialize_run
from kvresearch.navigator import Navigator, _cycle_budget_reached
from kvresearch.pipeline import _classify_smoke_failure, _implementation_smoke, _smoke_rollover_evidence
from kvresearch.util import atomic_write_json, atomic_write_jsonl, read_json, sha256_file
from tests.helpers import architecture_candidates, test_config, write_control, write_literature


class RepairRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = test_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_goal_mode_can_disable_operational_cycle_ceiling(self):
        self.assertTrue(_cycle_budget_reached(12, 11))
        with patch.dict("os.environ", {"KVRESEARCH_MAX_CYCLES_OVERRIDE": "unbounded"}):
            self.assertFalse(_cycle_budget_reached(12, 11))
            self.assertFalse(_cycle_budget_reached(12, 10_000))
        with patch.dict("os.environ", {"KVRESEARCH_MAX_CYCLES_OVERRIDE": "20"}):
            self.assertFalse(_cycle_budget_reached(12, 11))
            self.assertTrue(_cycle_budget_reached(12, 19))

    def test_smoke_requires_real_compressed_history_per_task(self):
        attempt = self.root / "attempt"
        atomic_write_jsonl(attempt / "raw" / "gsm8k.jsonl", [{"prefill_tokens": 120}])
        atomic_write_json(
            attempt / "diagnostics" / "gsm8k.json",
            {"observed_checkpoint_count": 0, "not_applicable_checkpoint_count": 3},
        )
        failed = _smoke_rollover_evidence(attempt, ["gsm8k"])
        self.assertFalse(failed["passed"])
        atomic_write_jsonl(attempt / "raw" / "gsm8k.jsonl", [{"prefill_tokens": 192}])
        atomic_write_json(
            attempt / "diagnostics" / "gsm8k.json",
            {"observed_checkpoint_count": 3, "not_applicable_checkpoint_count": 0},
        )
        passed = _smoke_rollover_evidence(attempt, ["gsm8k"])
        self.assertTrue(passed["passed"])

    def test_content_addressed_smoke_serializes_concurrent_validators(self):
        cycle = self.root / "run" / "cycles" / "000"
        implementation = cycle / "implementation_manifest.json"
        implementation.parent.mkdir(parents=True)
        implementation.write_text("{}\n", encoding="utf-8")
        state = {"active": 0, "peak": 0, "calls": 0}
        state_lock = threading.Lock()

        def fake_locked(*_args):
            with state_lock:
                state["active"] += 1
                state["calls"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with state_lock:
                state["active"] -= 1
            return {"passed": True}

        with patch("kvresearch.pipeline._implementation_smoke_locked", side_effect=fake_locked):
            threads = [
                threading.Thread(
                    target=_implementation_smoke,
                    args=(self.config, cycle.parents[1], cycle, {}, implementation),
                )
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(state["calls"], 2)
        self.assertEqual(state["peak"], 1)

    def _cycle(self, run_id: str = "repair-run"):
        run = initialize_run(self.config, "repair technical failures in place", run_id)
        write_literature(run / "evidence" / "literature.jsonl")
        write_control(run / "evidence" / "synthesis.json", "evidence_synthesis", run.name,
                      known_methods=[], bottlenecks=[], reusable_mechanisms=[], rejected_directions=[])
        cycle = run / "cycles" / "000"
        write_control(cycle / "problem_diagnosis.json", "problem_diagnosis", run.name,
                      bottleneck="x", observations=[], evidence=[], error_sources=[], falsification="f", exclusions=[])
        atomic_write_jsonl(cycle / "candidates.jsonl", architecture_candidates())
        write_control(cycle / "selection.json", "selection", run.name,
                      selected_candidate_id="c1", ranking=["c1", "c2", "c3", "c4"], rejected=[], evidence=[])
        workspace = cycle / "workspace"
        workspace.mkdir(parents=True)
        (workspace / ".git").mkdir()
        (workspace / "x.py").write_text("x=1\n", encoding="utf-8")
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name,
                      candidate_id="c1", code_scope=["x.py"], evaluation_tier="development",
                      datasets=["qasper", "lcc", "gov_report", "gsm8k", "math500"], decoding={"math_max_total_length": 2048},
                      outputs=[], budgets={}, decision_rules={},
                      storage_accounting={"theoretical_average_bits": 1.0, "estimated_actual_not_above_vecinfer": True},
                      workspace=str(workspace), execution={"adapter": "vecinfer"})
        write_control(cycle / "implementation_manifest.json", "implementation_manifest", run.name,
                      workspace=str(workspace), base_hash="a", changed_files=["x.py"], patch_hash="b",
                      build_command=[], run_entrypoint="scripts.modeldb.main_pq", dependencies=[])
        return run, cycle

    def _passing_validation(self, cycle: Path, run_id: str):
        write_control(cycle / "validation.json", "validation", run_id,
                      syntax={}, imports={}, interfaces={}, model_smoke={"passed": True}, data_hashes={},
                      protocol_match={}, theoretical_storage={}, estimated_storage={}, all_hard_checks_pass=True)

    def _attempt(self, cycle: Path, run_id: str, exit_code: int, status: str, error_code=None):
        attempt = cycle / "attempts" / "000"
        write_control(attempt / "preflight.json", "preflight", run_id, host="h", captured_at="x",
                      gpu_index=0, gpu_uuid="u", gpu_memory={}, gpu_load={}, processes=[], software={}, disk={},
                      approved_gpus=[{"index": 0, "uuid": "u"}], requested_gpu_count=1,
                      requested_resources_available=True)
        write_control(attempt / "run_manifest.json", "attempt_run_manifest", run_id, command=[],
                      cwd=str(cycle / "workspace"), env_allowlist=[],
                      input_hashes={"implementation": sha256_file(cycle / "implementation_manifest.json")},
                      seeds=[42], shards=[], started_at="x", timeout=1, pid=1)
        write_control(attempt / "exit.json", "attempt_exit", run_id, exit_code=exit_code, ended_at="x",
                      elapsed_seconds=1, timed_out=error_code == "E_TIMEOUT", oom=error_code == "E_GPU_OOM",
                      completed_tasks=[] if exit_code else ["qasper"], missing_tasks=[] if not exit_code else ["qasper"],
                      log_hashes={}, status=status, error_code=error_code)
        return attempt

    def test_failed_real_smoke_routes_to_same_cycle_implementation_repair(self):
        run, cycle = self._cycle("smoke-repair")
        write_control(cycle / "validation.json", "validation", run.name,
                      syntax={}, imports={}, interfaces={}, model_smoke={"passed": False}, data_hashes={},
                      protocol_match={}, theoretical_storage={}, estimated_storage={},
                      all_hard_checks_pass=False, status="failed", error_code="E_IMPLEMENTATION_NOT_READY")
        action = Navigator(self.config).next(run)
        self.assertEqual((action.kind, action.step), ("agent", "implementation_repair"))
        self.assertEqual(Path(action.output), cycle / "implementation_manifest.json")

    def test_nan_probability_trace_is_not_misclassified_as_missing_codebook(self):
        stderr = """
        File \"modeling_llama.py\", line 1, in load_centroid_codebook
        RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
        CUDA error: device-side assert triggered
        """
        self.assertEqual(
            _classify_smoke_failure(stderr, []),
            "candidate_numerical_instability",
        )

    def test_numerical_smoke_failure_routes_to_same_cycle_without_research_decision(self):
        run, cycle = self._cycle("numerical-smoke-repair")
        write_control(cycle / "validation.json", "validation", run.name,
                      syntax={}, imports={}, interfaces={},
                      model_smoke={
                          "passed": False,
                          "failure_class": "implementation_not_ready",
                          "repairable": True,
                          "tasks": [{
                              "task": "gsm8k",
                              "passed": False,
                              "failure_class": "candidate_numerical_instability",
                          }],
                      },
                      data_hashes={}, protocol_match={}, theoretical_storage={}, estimated_storage={},
                      all_hard_checks_pass=False, status="failed",
                      error_code="E_IMPLEMENTATION_NOT_READY")
        action = Navigator(self.config).next(run)
        self.assertEqual((action.kind, action.step), ("agent", "implementation_repair"))
        self.assertEqual(Path(action.output), cycle / "implementation_manifest.json")
        self.assertEqual(Path(action.output).parent.name, "000")

    def test_execution_error_repairs_same_candidate_without_next_cycle(self):
        run, cycle = self._cycle("execution-repair")
        self._passing_validation(cycle, run.name)
        self._attempt(cycle, run.name, 1, "failed", "E_EXECUTION")
        action = Navigator(self.config).next(run)
        self.assertEqual((action.kind, action.step), ("agent", "execution_repair"))
        self.assertEqual(Path(action.output), cycle / "implementation_manifest.json")

    def test_repaired_implementation_never_reuses_stale_attempt_directory(self):
        run, cycle = self._cycle("fresh-attempt-after-repair")
        self._passing_validation(cycle, run.name)
        attempt = self._attempt(cycle, run.name, 0, "complete")

        # Reproduce the real repair state: the plan-bound preflight is stale
        # and the old run manifest names a different implementation hash.
        preflight = read_json(attempt / "preflight.json")
        preflight["inputs"] = [{
            "path": str(cycle / "experiment_plan.json"),
            "sha256": "0" * 64,
        }]
        atomic_write_json(attempt / "preflight.json", preflight)
        write_control(cycle / "implementation_manifest.json", "implementation_manifest", run.name,
                      workspace=str(cycle / "workspace"), base_hash="a", changed_files=["x.py"],
                      patch_hash="repaired", build_command=[],
                      run_entrypoint="scripts.modeldb.main_pq", dependencies=[])

        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "preflight")
        self.assertEqual(Path(action.output), cycle / "attempts" / "001" / "preflight.json")
        self.assertIn(str(attempt / "run_manifest.json"), action.inputs)

    def test_technical_repairs_continue_beyond_research_attempt_budget(self):
        run, cycle = self._cycle("technical-attempts-unbounded")
        self._passing_validation(cycle, run.name)
        first = self._attempt(cycle, run.name, 1, "failed", "E_EXECUTION")
        old_hash = read_json(first / "run_manifest.json")["input_hashes"]["implementation"]
        for index in (1, 2):
            attempt = cycle / "attempts" / f"{index:03d}"
            write_control(attempt / "preflight.json", "preflight", run.name, host="h", captured_at="x",
                          gpu_index=0, gpu_uuid="u", gpu_memory={}, gpu_load={}, processes=[], software={}, disk={},
                          approved_gpus=[{"index": 0, "uuid": "u"}], requested_gpu_count=1,
                          requested_resources_available=True)
            write_control(attempt / "run_manifest.json", "attempt_run_manifest", run.name, command=[],
                          cwd=str(cycle / "workspace"), env_allowlist=[],
                          input_hashes={"implementation": old_hash}, seeds=[42], shards=[],
                          started_at="x", timeout=1, pid=1)
            write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=1, ended_at="x",
                          elapsed_seconds=1, timed_out=False, oom=False, completed_tasks=[],
                          missing_tasks=["qasper"], log_hashes={}, status="failed",
                          error_code="E_EXECUTION")
        write_control(cycle / "implementation_manifest.json", "implementation_manifest", run.name,
                      workspace=str(cycle / "workspace"), base_hash="a", changed_files=["x.py"],
                      patch_hash="third-repair", build_command=[],
                      run_entrypoint="scripts.modeldb.main_pq", dependencies=[])

        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "preflight")
        self.assertEqual(Path(action.output), cycle / "attempts" / "003" / "preflight.json")

    def test_incomplete_metrics_repair_same_candidate(self):
        run, cycle = self._cycle("metrics-repair")
        self._passing_validation(cycle, run.name)
        attempt = self._attempt(cycle, run.name, 0, "complete")
        write_control(cycle / "metrics.json", "metrics", run.name, tasks={}, longbench={"score": None},
                      gsm8k=None, math500=None, theoretical_storage_pass=False, actual_storage_pass=False,
                      storage_accounting_summary={"records": 0, "missing_actual_bytes": 1, "over_vecinfer_budget": 0},
                      peak_gpu_memory=0, elapsed_seconds=1, failures=["missing predictions"],
                      comparison_mode="development_candidate_ranking", evaluation_coverage={}, status="failed")
        action = Navigator(self.config).next(run)
        self.assertEqual((action.kind, action.step), ("agent", "execution_repair"))
        self.assertIn(str(attempt / "exit.json"), action.inputs)

    def test_measured_bit_excess_is_research_result_not_technical_repair(self):
        run, cycle = self._cycle("bit-failure")
        self._passing_validation(cycle, run.name)
        self._attempt(cycle, run.name, 0, "complete")
        write_control(cycle / "metrics.json", "metrics", run.name, tasks={}, longbench={"score": 0},
                      gsm8k={"score": 0}, math500={"score": 0}, theoretical_storage_pass=True,
                      actual_storage_pass=False,
                      storage_accounting_summary={"records": 5, "missing_actual_bytes": 0, "over_vecinfer_budget": 1},
                      peak_gpu_memory=0, elapsed_seconds=1, failures=[],
                      comparison_mode="development_candidate_ranking", evaluation_coverage={})
        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "verify")


if __name__ == "__main__":
    unittest.main()
