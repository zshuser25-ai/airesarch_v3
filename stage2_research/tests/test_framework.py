from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kvresearch.initialize import initialize_run
from kvresearch.navigator import Navigator
from kvresearch.pipeline import aggregate, decide
from kvresearch.schema import ArtifactValidator
from kvresearch.storage import vecinfer_reference_bytes
from kvresearch.util import atomic_write_json, atomic_write_jsonl, read_json, sha256_file
from tests.helpers import architecture_candidates, test_config, write_control, write_literature


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = test_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_initialize_and_first_route(self):
        run = initialize_run(self.config, "improve one-bit KV cache", "test-run")
        self.assertTrue((run / "inputs.lock.json").is_file())
        validator = ArtifactValidator(self.config.catalog_path)
        self.assertTrue(validator.validate_control(run / "inputs.lock.json", "inputs_lock").valid)
        action = Navigator(self.config).next(run)
        self.assertEqual((action.kind, action.step), ("agent", "literature"))
        self.assertFalse((run / "cycles").exists(), "next must be read-only")

    def test_diagnosis_evidence_excludes_stale_structured_paths_and_hashes(self):
        run = self.root / "probe-run"
        evidence = run / "evidence"
        current = run / "cycles" / "009" / "metrics.json"
        atomic_write_json(current, {"status": "complete"})
        current_sha = sha256_file(current)
        atomic_write_json(
            evidence / "cycle009_current.json",
            {"status": "complete", "input": {"path": str(current), "sha256": current_sha}},
        )
        atomic_write_json(
            evidence / "cycle009_missing.json",
            {"status": "complete", "inputs": {"candidate": str(run / "cycles/009/old-candidate.jsonl")}},
        )
        atomic_write_json(
            evidence / "cycle009_bad_hash.json",
            {"status": "complete", "input": {"path": str(current), "sha256": "0" * 64}},
        )
        self.assertEqual(
            Navigator._diagnosis_evidence(run),
            (evidence / "cycle009_current.json",),
        )

    def test_restarted_cycle_diagnosis_keeps_previous_lessons(self):
        run = initialize_run(self.config, "improve one-bit KV cache", "restart-diagnosis-run")
        write_literature(run / "evidence" / "literature.jsonl")
        write_control(
            run / "evidence" / "synthesis.json", "evidence_synthesis", run.name,
            known_methods=[], bottlenecks=[], reusable_mechanisms=[], rejected_directions=[],
        )
        previous = run / "cycles" / "000"
        write_control(
            previous / "decision.json", "decision", run.name,
            action="REFINE", verified_value_ids=[], rationale="continue",
            next_cycle_constraints=[], cooldown={}, measurement_valid=True,
        )
        atomic_write_jsonl(previous / "lessons.jsonl", [{
            "lesson_id": "lesson-000", "source_cycle": "000", "decision": "REFINE",
            "scope": "test", "failed_mechanism": "none", "effective_fix": "retain evidence",
            "evidence": [str(previous / "decision.json")], "confidence": 1.0,
        }])
        (run / "cycles" / "001").mkdir(parents=True)

        action = Navigator(self.config).next(run)

        self.assertEqual(action.step, "problem_diagnosis")
        self.assertIn(str(previous / "decision.json"), action.inputs)
        self.assertIn(str(previous / "lessons.jsonl"), action.inputs)
        self.assertIn(str(run / "inputs.lock.json"), action.inputs)

    def test_development_promote_routes_directly_to_formal_plan(self):
        run = initialize_run(self.config, "improve one-bit KV cache", "formal-promotion-run")
        write_literature(run / "evidence" / "literature.jsonl")
        write_control(
            run / "evidence" / "synthesis.json", "evidence_synthesis", run.name,
            known_methods=[], bottlenecks=[], reusable_mechanisms=[], rejected_directions=[],
        )
        previous = run / "cycles" / "000"
        workspace = previous / "workspace"
        write_control(
            previous / "selection.json", "selection", run.name,
            selected_candidate_id="promoted", ranking=["promoted", "c2", "c3", "c4"],
            rejected=[], evidence=[],
        )
        write_control(
            previous / "experiment_plan.json", "experiment_plan", run.name,
            candidate_id="promoted", code_scope=["scripts"], evaluation_tier="development",
            datasets=["qasper", "lcc", "gov_report", "gsm8k", "math500"],
            decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={},
            storage_accounting={"theoretical_average_bits": 1.0, "estimated_actual_not_above_vecinfer": True},
            workspace=str(workspace), execution={"adapter": "vecinfer"},
        )
        write_control(
            previous / "implementation_manifest.json", "implementation_manifest", run.name,
            workspace=str(workspace), base_hash="base", changed_files=["scripts/x.py"],
            patch_hash="patch", build_command=[], run_entrypoint="scripts.modeldb.main_pq", dependencies=[],
        )
        write_control(
            previous / "validation.json", "validation", run.name,
            syntax={}, imports={}, interfaces={}, model_smoke={}, data_hashes={}, protocol_match={},
            theoretical_storage={}, estimated_storage={}, all_hard_checks_pass=True,
        )
        write_control(
            previous / "decision.json", "decision", run.name,
            action="PROMOTE", verified_value_ids=[], rationale="development passed",
            next_cycle_constraints=[], cooldown={}, measurement_valid=True,
        )
        (run / "cycles" / "001").mkdir(parents=True)

        action = Navigator(self.config).next(run)

        self.assertEqual((action.kind, action.step), ("agent", "experiment_plan"))
        self.assertEqual(action.output, str(run / "cycles" / "001" / "experiment_plan.json"))
        self.assertIn(str(previous / "decision.json"), action.inputs)
        self.assertIn(str(previous / "implementation_manifest.json"), action.inputs)

    def test_experiment_plan_rejects_workspace_object_and_missing_global_math_cap(self):
        validator = ArtifactValidator(self.config.catalog_path)
        plan_path = self.root / "bad-plan.json"
        write_control(
            plan_path, "experiment_plan", "bad-plan-run",
            candidate_id="c1", code_scope=["x.py"], evaluation_tier="development",
            datasets=["qasper"], decoding={"gsm8k": {"max_total_length": 2048}},
            outputs=[], budgets={}, decision_rules={}, storage_accounting={},
            workspace={"path": str(self.root / "workspace")}, execution={"adapter": "vecinfer"},
        )
        result = validator.validate_control(plan_path, "experiment_plan")
        self.assertFalse(result.valid)
        self.assertTrue(any("workspace must be" in error for error in result.errors))
        self.assertTrue(any("decoding.math_max_total_length" in error for error in result.errors))

    def test_experiment_plan_rejects_math_cap_applied_to_longbench(self):
        validator = ArtifactValidator(self.config.catalog_path)
        plan_path = self.root / "bad-longbench-cap-plan.json"
        write_control(
            plan_path, "experiment_plan", "bad-longbench-cap-run",
            candidate_id="c1", code_scope=["x.py"], evaluation_tier="development",
            datasets=["qasper", "gsm8k", "math500"],
            decoding={
                "math_max_total_length": 2048,
                "longbench": {"method": "greedy", "max_total_length": 2048},
                "gsm8k": {"max_total_length": 2048},
                "math500": {"max_total_length": 2048},
            },
            outputs=[], budgets={}, decision_rules={}, storage_accounting={},
            workspace=str(self.root / "workspace"), execution={"adapter": "vecinfer"},
        )
        result = validator.validate_control(plan_path, "experiment_plan")
        self.assertFalse(result.valid)
        self.assertTrue(any("must not apply" in error for error in result.errors))

    def test_routing_to_preflight_execute_and_monitor(self):
        run = initialize_run(self.config, "improve one-bit KV cache", "route-run")
        write_literature(run / "evidence" / "literature.jsonl")
        write_control(run / "evidence" / "synthesis.json", "evidence_synthesis", run.name, known_methods=[], bottlenecks=[], reusable_mechanisms=[], rejected_directions=[])
        cycle = run / "cycles" / "000"
        write_control(cycle / "problem_diagnosis.json", "problem_diagnosis", run.name, bottleneck="x", observations=[], evidence=[], error_sources=[], falsification="test", exclusions=[])
        atomic_write_jsonl(cycle / "candidates.jsonl", architecture_candidates())
        write_control(cycle / "selection.json", "selection", run.name, selected_candidate_id="c1", ranking=["c1", "c2", "c3", "c4"], rejected=[], evidence=[])
        workspace = cycle / "workspace"
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"], evaluation_tier="development", datasets=["qasper", "lcc", "gov_report", "gsm8k", "math500"], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={}, storage_accounting={"theoretical_average_bits": 1.0, "estimated_actual_not_above_vecinfer": True}, workspace=str(workspace), execution={"adapter": "vecinfer"})
        write_control(cycle / "implementation_manifest.json", "implementation_manifest", run.name, workspace=str(workspace), base_hash="a", changed_files=["x.py"], patch_hash="b", build_command=[], run_entrypoint="x", dependencies=[])
        write_control(cycle / "validation.json", "validation", run.name, syntax={}, imports={}, interfaces={}, model_smoke={}, data_hashes={}, protocol_match={}, theoretical_storage={}, estimated_storage={}, all_hard_checks_pass=True)
        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "preflight")
        attempt = cycle / "attempts" / "000"
        contaminated = attempt / "task_build" / "manual-probe"
        contaminated.mkdir(parents=True)
        guarded = Navigator(self.config).next(run)
        self.assertEqual(guarded.step, "preflight")
        self.assertIn("attempts/001/preflight.json", guarded.output)
        contaminated.rmdir()
        (attempt / "task_build").rmdir()
        attempt.rmdir()
        write_control(attempt / "preflight.json", "preflight", run.name, host="h", captured_at="x", gpu_index=0, gpu_uuid="u", gpu_memory={}, gpu_load={}, processes=[], software={}, disk={}, approved_gpus=[], requested_gpu_count=1, requested_resources_available=False)
        action = Navigator(self.config).next(run)
        self.assertEqual((action.state, action.step, action.kind), ("WAIT", "remote_resources", "terminal"))
        with patch.dict("os.environ", {"KVRESEARCH_RESUME_REMOTE_ONCE": "1"}):
            resumed = Navigator(self.config).next(run)
        self.assertEqual(resumed.step, "preflight")
        self.assertIn("attempts/001/preflight.json", resumed.output)
        write_control(attempt / "preflight.json", "preflight", run.name, host="h", captured_at="x", gpu_index=0, gpu_uuid="u", gpu_memory={}, gpu_load={}, processes=[], software={}, disk={}, approved_gpus=[{"index": 0, "uuid": "u"}], requested_gpu_count=1, requested_resources_available=True)
        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "execute")
        write_control(attempt / "run_manifest.json", "attempt_run_manifest", run.name, command=[], cwd=str(workspace), env_allowlist=[], input_hashes={"implementation": sha256_file(cycle / "implementation_manifest.json")}, seeds=[42], shards=[], started_at="x", timeout=1, pid=999999, status="partial")
        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "monitor")
        write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=124, ended_at="x", elapsed_seconds=1,
                      timed_out=True, oom=False, completed_tasks=[], missing_tasks=["qasper"], log_hashes={},
                      status="failed", error_code="E_TIMEOUT")
        action = Navigator(self.config).next(run)
        self.assertEqual(action.step, "execution_repair")
        self.assertEqual(action.kind, "agent")

        # A host-only executor repair must allocate a fresh attempt without
        # forcing an unrelated candidate implementation rewrite.
        exit_path = attempt / "exit.json"
        exit_path.unlink()
        write_control(
            attempt / "run_manifest.json", "attempt_run_manifest", run.name,
            command=[], cwd=str(workspace), env_allowlist=[],
            input_hashes={
                "implementation": sha256_file(cycle / "implementation_manifest.json"),
                "host_executor": "stale-host-executor",
            },
            seeds=[42], shards=[], started_at="x", timeout=1, pid=999999,
            status="partial",
        )
        host_repaired = Navigator(self.config).next(run)
        self.assertEqual(host_repaired.step, "preflight")
        self.assertIn("attempts/001/preflight.json", host_repaired.output)

    def test_storage_reference_and_longbench_aggregate(self):
        run = initialize_run(self.config, "test aggregation", "aggregate-run")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        plan = write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"], evaluation_tier="development", datasets=["qasper"], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={}, storage_accounting={"theoretical_average_bits": 1.0}, workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"})
        write_control(cycle / "validation.json", "validation", run.name, syntax={}, imports={}, interfaces={}, model_smoke={}, data_hashes={}, protocol_match={}, theoretical_storage={}, estimated_storage={}, all_hard_checks_pass=True)
        write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=0, ended_at="x", elapsed_seconds=1, timed_out=False, oom=False, completed_tasks=["qasper"], missing_tasks=[], log_hashes={})
        reference = vecinfer_reference_bytes(
            self.config,
            1024,
            16,
            prefill_tokens=1000,
            output_tokens=24,
        )
        rows = []
        for index in range(32):
            rows.append({"sample_idx": index, "dataset": "qasper", "prompt": "p", "prediction": "a", "answers": ["a"], "score": 50.0, "prefill_tokens": 1000, "output_tokens": 24, "total_tokens": 1024, "actual_pq_kv_cache_bytes": reference})
        atomic_write_jsonl(attempt / "raw" / "qasper.jsonl", rows)
        result = aggregate(self.config, run, "000")
        self.assertEqual(result["tasks"]["qasper"]["samples"], 32)
        self.assertTrue(result["tasks"]["qasper"]["sample_ids_match"])
        self.assertTrue(result["actual_storage_pass"])
        self.assertEqual(result["longbench"]["score"], 50.0)

    def test_storage_reference_matches_vecinfer_prefill_decode_flushes(self):
        # LongBench sink=16 uses 112-token prefill chunks, while a decode-time
        # flush writes the complete 128-row arena.  The last generated token is
        # not inserted into the cache.
        self.assertEqual(
            vecinfer_reference_bytes(
                self.config,
                2714,
                16,
                prefill_tokens=2696,
                output_tokens=18,
            ),
            38_928_384,
        )
        self.assertEqual(
            vecinfer_reference_bytes(
                self.config,
                6849,
                16,
                prefill_tokens=6820,
                output_tokens=29,
            ),
            71_827_456,
        )

    def test_matched_join_is_authoritative_over_natural_row_lengths(self):
        run = initialize_run(self.config, "matched storage", "matched-storage-run")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        write_control(
            cycle / "experiment_plan.json", "experiment_plan", run.name,
            candidate_id="c1", code_scope=["x.py"], evaluation_tier="development",
            datasets=["qasper"], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={},
            storage_accounting={"theoretical_average_bits": 1.0},
            workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"},
        )
        write_control(cycle / "validation.json", "validation", run.name, all_hard_checks_pass=True)
        write_control(
            attempt / "exit.json", "attempt_exit", run.name,
            exit_code=0, ended_at="x", elapsed_seconds=1, timed_out=False,
            oom=False, completed_tasks=["qasper"], missing_tasks=[], log_hashes={},
        )
        rows = [
            {
                "sample_idx": index, "dataset": "qasper", "prompt": "p",
                "prediction": "a", "answers": ["a"], "score": 50.0,
                "prefill_tokens": 1000, "output_tokens": 24,
                "total_tokens": 1024,
                # Deliberately above the legacy natural-length formula.  This
                # value includes fixed codebooks/workspace and is not the hard
                # bit comparator for a paired implementation.
                "actual_pq_kv_cache_bytes": 999_999_999,
            }
            for index in range(32)
        ]
        atomic_write_jsonl(attempt / "raw" / "qasper.jsonl", rows)
        control_path = attempt / "arms" / "qasper" / "unmodified_vecinfer.jsonl"
        atomic_write_jsonl(control_path, rows)
        atomic_write_json(
            attempt / "joins" / "qasper.json",
            {
                "complete": True,
                "arms": {
                    "unmodified_vecinfer": {
                        "path": str(control_path),
                        "sha256": sha256_file(control_path),
                    }
                },
                "storage_gate_pass": True,
                "matched_storage": {
                    "comparison": {
                        "complete": True,
                        "logical_length": 16384,
                        "candidate_no_greater_than_vecinfer": True,
                        "charged_scopes": {
                            "actual_pq_kv_cache_bytes": {
                                "control": 153_024_344,
                                "candidate": 152_500_000,
                                "delta": -524_344,
                            },
                            "persistent_kv_related_bytes": {
                                "control": 152_000_000,
                                "candidate": 151_500_000,
                                "delta": -500_000,
                            },
                            "peak_kv_related_bytes": {
                                "control": 153_024_344,
                                "candidate": 152_500_000,
                                "delta": -524_344,
                            },
                            # An internal component may grow when the complete
                            # charged allocation still shrinks.
                            "routing_index_bytes": {
                                "control": 1_000_000,
                                "candidate": 1_057_856,
                                "delta": 57_856,
                            },
                        },
                    }
                },
            },
        )

        result = aggregate(self.config, run, "000")

        self.assertTrue(result["actual_storage_pass"])
        self.assertEqual(
            result["storage_accounting_summary"]["measurement_basis"],
            "matched_fixed_length_join",
        )
        self.assertEqual(result["storage_accounting_summary"]["over_vecinfer_budget"], 0)

    def test_formal_success_stops(self):
        run = initialize_run(self.config, "test decision", "decision-run")
        cycle = run / "cycles" / "000"
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"], evaluation_tier="formal", datasets=[], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={}, storage_accounting={}, workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"})
        metrics = write_control(cycle / "metrics.json", "metrics", run.name, tasks={}, longbench={"score": 46.0}, gsm8k={"score": 3.0}, math500={"score": 4.0}, math={"score": 3.5}, theoretical_storage_pass=True, actual_storage_pass=True, peak_gpu_memory=0, elapsed_seconds=0, failures=[], comparison_mode="full_30plus70_vecinfer_1bit", baseline_reference={"longbench": {"score": 45.5}, "gsm8k": {"score": 0.0}, "math500": {"score": 2.8}})
        registry = write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})
        write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name, deficiencies=[], severity="info", evidence=[], repairable=False, suggested_actions=[])
        result = decide(self.config, run, "000")
        self.assertEqual(result["action"], "STOP")

    def test_formal_none_better_archives(self):
        run = initialize_run(self.config, "test formal failure", "formal-fail")
        cycle = run / "cycles" / "000"
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"], evaluation_tier="formal", datasets=[], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={}, storage_accounting={}, workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"})
        write_control(cycle / "metrics.json", "metrics", run.name, tasks={}, longbench={"score": 45.0}, gsm8k={"score": 0.0}, math500={"score": 2.0}, math={"score": 1.0}, theoretical_storage_pass=True, actual_storage_pass=True, peak_gpu_memory=0, elapsed_seconds=0, failures=[], comparison_mode="paired_fixed_30pct_vecinfer_1bit", baseline_reference={"longbench": {"score": 45.5}, "gsm8k": {"score": 0.0}, "math500": {"score": 2.8}})
        write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})
        write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name, deficiencies=[], severity="info", evidence=[], repairable=False, suggested_actions=[])
        result = decide(self.config, run, "000")
        self.assertEqual(result["action"], "ARCHIVE")
        self.assertFalse(any(result["comparisons"]["formal_improvements"].values()))

    def test_fixed_sample_identity_mismatch_fails_aggregation(self):
        run = initialize_run(self.config, "test fixed identities", "identity-run")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"], evaluation_tier="development", datasets=["qasper"], decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={}, storage_accounting={"theoretical_average_bits": 1.0}, workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"})
        write_control(cycle / "validation.json", "validation", run.name, syntax={}, imports={}, interfaces={}, model_smoke={}, data_hashes={}, protocol_match={}, theoretical_storage={}, estimated_storage={}, all_hard_checks_pass=True)
        write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=0, ended_at="x", elapsed_seconds=1, timed_out=False, oom=False, completed_tasks=["qasper"], missing_tasks=[], log_hashes={})
        rows = [{"sample_idx": index + 1, "dataset": "qasper", "prediction": "a", "answers": ["a"], "score": 1.0,
                 "total_tokens": 256, "actual_pq_kv_cache_bytes": vecinfer_reference_bytes(self.config, 256, 16)} for index in range(32)]
        atomic_write_jsonl(attempt / "raw" / "qasper.jsonl", rows)
        result = aggregate(self.config, run, "000")
        self.assertFalse(result["tasks"]["qasper"]["sample_ids_match"])
        self.assertEqual(result["status"], "failed")

    def test_formal_30_and_70_are_combined_by_sample(self):
        run = initialize_run(self.config, "test full formal merge", "merge-run")
        cycle = run / "cycles" / "000"
        attempt = cycle / "attempts" / "000"
        tasks = ["qasper", "gsm8k", "math500"]
        write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"],
                      evaluation_tier="formal", datasets=tasks, decoding={"math_max_total_length": 2048}, outputs=[], budgets={}, decision_rules={},
                      storage_accounting={"theoretical_average_bits": 1.0}, workspace=str(cycle / "workspace"), execution={"adapter": "vecinfer"})
        write_control(cycle / "validation.json", "validation", run.name, all_hard_checks_pass=True)
        write_control(attempt / "exit.json", "attempt_exit", run.name, exit_code=0, ended_at="x", elapsed_seconds=1,
                      timed_out=False, oom=False, completed_tasks=[], missing_tasks=[], log_hashes={})
        from kvresearch.util import atomic_write_json
        atomic_write_json(attempt / "formal_30pct_gate.json", {"qualified_for_remaining_70pct": True})
        for task in tasks:
            for raw_name, count in (("raw", 3), ("raw_remaining", 7)):
                rows = []
                for index in range(count):
                    reference = vecinfer_reference_bytes(self.config, 256, 0 if task in {"gsm8k", "math500"} else 16)
                    rows.append({"sample_idx": index, "score": 50.0, "total_tokens": 256,
                                 "actual_pq_kv_cache_bytes": reference, "prediction": "x", "answers": ["x"]})
                atomic_write_jsonl(attempt / raw_name / f"{task}.jsonl", rows)

        def fake_math(config, attempt_dir, task, raw_path, phase="combined"):
            count = len(__import__("kvresearch.util", fromlist=["read_jsonl"]).read_jsonl(raw_path))
            return ([{"score": [True], "code": "x", "gt": "x"} for _ in range(count)], {})

        with patch("kvresearch.pipeline._math_eval", side_effect=fake_math):
            result = aggregate(self.config, run, "000")
        self.assertEqual(result["comparison_mode"], "full_30plus70_vecinfer_1bit")
        self.assertTrue(result["evaluation_coverage"]["full_dataset"])
        self.assertEqual(result["tasks"]["qasper"]["samples"], 10)
        self.assertTrue(result["tasks"]["qasper"]["sample_ids_match"])


if __name__ == "__main__":
    unittest.main()
