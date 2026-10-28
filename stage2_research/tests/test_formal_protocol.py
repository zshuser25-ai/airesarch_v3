from __future__ import annotations

import tempfile
import unittest
import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

from kvresearch.executor import (
    _formal_30pct_gate,
    _effective_batch_attempts,
    _run_dynamic_gpu_pool,
    _sample_shard_specs,
    _wait_for_committed_file,
)
from kvresearch.initialize import initialize_run
from kvresearch.navigator import _artifact_binds_current_inputs
from kvresearch.pipeline import _math_budget_check, decide, diagnose, learn
from kvresearch.remote import RemoteCleanupError, RemotePod
from kvresearch.util import atomic_write_jsonl, read_jsonl
from kvresearch.vecinfer import CommandSpec, VecInferAdapter
from kvresearch.stock_control import stock_codebook_manifest
from tests.helpers import test_config, write_control


class FormalProtocolTests(unittest.TestCase):
    def test_atomic_commit_wait_tolerates_delayed_nfs_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            commit = Path(tmp) / "joins" / "task.json"

            def publish():
                time.sleep(0.05)
                commit.parent.mkdir()
                commit.write_text('{"complete": true}\n', encoding="utf-8")

            publisher = threading.Thread(target=publish)
            publisher.start()
            digest = _wait_for_committed_file(commit, 1.0, poll_seconds=0.01)
            publisher.join()

            self.assertIsNotNone(digest)
            self.assertEqual(len(digest), 64)

    def test_atomic_commit_wait_is_bounded_when_artifact_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            digest = _wait_for_committed_file(
                Path(tmp) / "missing.json", 0.03, poll_seconds=0.005
            )

            self.assertIsNone(digest)
            self.assertLess(time.monotonic() - started, 0.5)

    def test_effective_batch_attempts_remove_redundant_capped_retries(self):
        self.assertEqual(_effective_batch_attempts([16, 8, 4, 2, 1], 4), [4, 2, 1])

    def test_formal_sample_shards_are_disjoint_and_probe_once_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            dataset_root.mkdir()
            (dataset_root / ".complete.json").write_text(json.dumps({
                "status": "complete",
                "manifest": {"tasks": {
                    "long": {"output": {"rows": 65}},
                    "short": {"output": {"rows": 7}},
                }},
            }), encoding="utf-8")
            specs = []
            for task in ("long", "short"):
                output = root / "attempt" / "raw" / f"{task}.jsonl"
                specs.append(CommandSpec(
                    f"generate-{task}",
                    ["python", "worker", "--batch_size", "16", "--save_path", str(output)],
                    root, {"KVRESEARCH_DATASETS_ROOT": str(dataset_root)}, 10, output,
                ))

            units, grouped = _sample_shard_specs(
                specs, root / "attempt", "initial_30pct", 30
            )

            self.assertEqual(
                [(unit.sample_start, unit.sample_end) for unit in grouped["long"]],
                [(0, 30), (30, 60), (60, 65)],
            )
            self.assertEqual(
                [(unit.sample_start, unit.sample_end) for unit in grouped["short"]],
                [(0, 7)],
            )
            self.assertEqual(len(units), 4)
            for task_units in grouped.values():
                self.assertEqual(
                    sum(unit.env["KVRESEARCH_SAMPLE_SHARD_PROBE_OWNER"] == "1" for unit in task_units),
                    1,
                )
                self.assertTrue(all(unit.task_name == task_units[0].task_name for unit in task_units))
                self.assertTrue(all(unit.final_output == task_units[0].final_output for unit in task_units))

    def test_math_sample_shards_use_smaller_dynamic_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets"
            dataset_root.mkdir()
            (dataset_root / ".complete.json").write_text(json.dumps({
                "status": "complete",
                "manifest": {"tasks": {
                    "long": {"output": {"rows": 65}},
                    "math500": {"output": {"rows": 65}},
                }},
            }), encoding="utf-8")
            specs = []
            for task in ("long", "math500"):
                output = root / "attempt" / "raw" / f"{task}.jsonl"
                specs.append(CommandSpec(
                    f"generate-{task}",
                    ["python", "worker", "--batch_size", "16", "--save_path", str(output)],
                    root, {"KVRESEARCH_DATASETS_ROOT": str(dataset_root)}, 10, output,
                ))

            _, grouped = _sample_shard_specs(
                specs, root / "attempt", "initial_30pct", 60,
                {"math500": 30},
            )

            self.assertEqual(
                [(unit.sample_start, unit.sample_end) for unit in grouped["long"]],
                [(0, 60), (60, 65)],
            )
            self.assertEqual(
                [(unit.sample_start, unit.sample_end) for unit in grouped["math500"]],
                [(0, 30), (30, 60), (60, 65)],
            )

    def test_remote_timeout_runs_token_scoped_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            pod = RemotePod(test_config(Path(tmp)))
            timeout = subprocess.TimeoutExpired(["ssh"], 10)
            cleanup_result = subprocess.CompletedProcess(["ssh"], 0, "CLEANUP_CONFIRMED\n", "")
            with patch.object(pod, "run", side_effect=[timeout, cleanup_result]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    pod.execute_argv(["python", "worker.py"], Path(tmp), {}, 10)

            launch_command = run.call_args_list[0].args[0]
            cleanup_command = run.call_args_list[1].args[0]
            self.assertIn("setsid env KVRESEARCH_EXECUTION_TOKEN=", launch_command)
            self.assertIn("timeout --signal=TERM --kill-after=10 10 sh -c", launch_command)
            self.assertIn("kvresearch-token-", launch_command)
            self.assertIn("kill -TERM -- \"-$pgid\"", cleanup_command)
            self.assertIn('"$proc/environ"', cleanup_command)
            self.assertIn('kill -0 -- "-$pgid"', cleanup_command)
            self.assertNotIn("pkill", cleanup_command)

    def test_validation_provenance_rejects_changed_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            implementation = root / "implementation.json"
            plan.write_text("plan-v1", encoding="utf-8")
            implementation.write_text("implementation-v1", encoding="utf-8")
            import hashlib
            validation = {
                "inputs": [
                    {"path": str(plan), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
                    {"path": str(implementation), "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest()},
                ]
            }
            self.assertTrue(_artifact_binds_current_inputs(validation, (plan, implementation)))
            implementation.write_text("implementation-v2", encoding="utf-8")
            self.assertFalse(_artifact_binds_current_inputs(validation, (plan, implementation)))

    def test_dynamic_gpu_pool_refills_the_first_available_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            third_started = threading.Event()
            calls = []

            class FakeRemote:
                def runtime_sample(self, gpu_uuid, label):
                    return {"gpu_uuid": gpu_uuid, "label": label}

                def execute_argv(self, argv, cwd, env, timeout):
                    task = Path(argv[argv.index("--save_path") + 1]).stem
                    calls.append((task, env["CUDA_VISIBLE_DEVICES"]))
                    if task == "slow":
                        if not third_started.wait(timeout=2):
                            raise AssertionError("the free GPU was not dynamically refilled")
                    elif task == "third":
                        third_started.set()
                    return subprocess.CompletedProcess(argv, 0, "", "")

            class FakeAdapter:
                remote = FakeRemote()

            specs = [
                CommandSpec(
                    f"generate-{task}",
                    ["python", "worker", "--batch_size", "4", "--save_path", str(root / f"{task}.jsonl")],
                    root, {"CUDA_VISIBLE_DEVICES": "stale"}, 10, root / f"{task}.jsonl",
                )
                for task in ("slow", "fast", "third")
            ]
            records = _run_dynamic_gpu_pool(
                FakeAdapter(), specs,
                [{"index": 0, "uuid": "gpu-0"}, {"index": 1, "uuid": "gpu-1"}],
                root, [4], "formal",
            )

            self.assertEqual({record["task"] for record in records}, {"slow", "fast", "third"})
            by_gpu = {}
            for task, gpu in calls:
                by_gpu.setdefault(gpu, []).append(task)
            slow_gpu = next(gpu for task, gpu in calls if task == "slow")
            refill_gpu = next(gpu for task, gpu in calls if task == "third")
            self.assertNotEqual(slow_gpu, refill_gpu)
            self.assertEqual(len(by_gpu[slow_gpu]), 1)
            self.assertEqual(len(by_gpu[refill_gpu]), 2)

    def test_sample_units_activate_all_eight_gpu_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            barrier = threading.Barrier(8)
            seen = set()
            lock = threading.Lock()

            class FakeRemote:
                def runtime_sample(self, gpu_uuid, label):
                    return {"gpu_uuid": gpu_uuid, "label": label}

                def execute_argv(self, argv, cwd, env, timeout):
                    gpu = env["CUDA_VISIBLE_DEVICES"]
                    first = False
                    with lock:
                        if gpu not in seen:
                            seen.add(gpu)
                            first = True
                    if first:
                        barrier.wait(timeout=2)
                    return subprocess.CompletedProcess(argv, 0, "", "")

            class FakeAdapter:
                remote = FakeRemote()

            specs = [
                CommandSpec(
                    f"generate-shared-{index}",
                    ["python", "worker", "--batch_size", "4", "--save_path", str(root / f"{index}.jsonl")],
                    root, {}, 10, root / f"{index}.jsonl",
                    task_name="shared", unit_id=f"shared:{index}",
                )
                for index in range(16)
            ]
            records = _run_dynamic_gpu_pool(
                FakeAdapter(), specs,
                [{"index": index, "uuid": f"gpu-{index}"} for index in range(8)],
                root, [4], "formal", journal_path=root / "scheduler_journal.jsonl",
            )

            self.assertEqual(len(records), 16)
            self.assertEqual(seen, {str(index) for index in range(8)})
            self.assertEqual(len(read_jsonl(root / "scheduler_journal.jsonl")), 16)

    def test_unconfirmed_cleanup_stops_same_gpu_from_claiming_more_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = []

            class FakeRemote:
                def runtime_sample(self, gpu_uuid, label):
                    return {"gpu_uuid": gpu_uuid, "label": label}

                def execute_argv(self, argv, cwd, env, timeout):
                    calls.append(env["KVRESEARCH_SAMPLE_SHARD_ID"])
                    raise RemoteCleanupError("E_REMOTE_CLEANUP: test")

                def cleanup_registry(self, registry_dir):
                    return False

            class FakeAdapter:
                remote = FakeRemote()

            specs = [
                CommandSpec(
                    f"generate-shared-{index}",
                    ["python", "worker", "--batch_size", "4", "--save_path", str(root / f"{index}.jsonl")],
                    root, {"KVRESEARCH_SAMPLE_SHARD_ID": f"shared:{index}"},
                    10, root / f"{index}.jsonl", task_name="shared", unit_id=f"shared:{index}",
                )
                for index in range(3)
            ]
            records = _run_dynamic_gpu_pool(
                FakeAdapter(), specs, [{"index": 0, "uuid": "gpu-0"}],
                root, [4], "formal",
            )

            self.assertEqual(calls, ["shared:0"])
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["cleanup_failed"])

    def test_development_preflight_honors_plan_gpu_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "multi-gpu development", "development-gpu-run")
            cycle = run / "cycles" / "000"
            write_control(
                cycle / "experiment_plan.json", "experiment_plan", run.name,
                evaluation_tier="development", execution={"adapter": "vecinfer", "gpu_count": 8},
            )
            gpu_rows = "\n".join(
                f"{index}, NVIDIA GeForce RTX 4090, GPU-{index}, 24564, 0, 24564, 0, 35, P8, 575.57.08"
                for index in range(8)
            ) + "\n"
            info = "\n".join([
                str(config.section("remote").get("expected_hostname") or "test-host"),
                "Linux test",
                json.dumps({"python": "3.11", "torch": "2.6", "runtime_imports": True}),
                "/dev/test 1 1 1 1% " + str(config.project_root),
            ]) + "\n"
            responses = [
                subprocess.CompletedProcess(["nvidia-smi"], 0, gpu_rows, ""),
                subprocess.CompletedProcess(["probe"], 0, info, ""),
            ]
            pod = RemotePod(config)
            with patch.object(pod, "run", side_effect=responses):
                result = pod.preflight(run, "000", "000")

            self.assertTrue(result["requested_resources_available"])
            self.assertEqual(result["requested_gpu_count"], 8)
            self.assertEqual(len(result["approved_gpus"]), 8)

    def test_zero_storage_records_are_unmeasured_not_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "missing storage", "missing-storage-run")
            cycle = run / "cycles" / "000"
            write_control(
                cycle / "metrics.json",
                "metrics",
                run.name,
                tasks={},
                theoretical_storage_pass=False,
                actual_storage_pass=False,
                storage_accounting_summary={"records": 0, "missing_actual_bytes": 0, "over_vecinfer_budget": 0},
                failures=["no closed execution attempt"],
                comparison_mode="development_candidate_ranking",
            )
            write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})

            result = diagnose(config, run, "000")
            kinds = {item["type"]: item for item in result["deficiencies"]}

            self.assertIn("theoretical_storage_unmeasured", kinds)
            self.assertFalse(kinds["theoretical_storage_unmeasured"]["counts_as_research_result"])
            self.assertIn("actual_storage_measurement_missing", kinds)
            self.assertNotIn("theoretical_storage_exceeded", kinds)
            self.assertNotIn("actual_storage_exceeded", kinds)

    def test_invalid_measurement_with_null_task_metrics_refines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "null metrics", "null-metrics-run")
            cycle = run / "cycles" / "000"
            atomic_write_jsonl(cycle / "candidates.jsonl", [{"candidate_id": "c", "mechanism_family": "family-a"}])
            write_control(cycle / "selection.json", "selection", run.name, selected_candidate_id="c", ranking=[], rejected=[], evidence=[])
            write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, evaluation_tier="development")
            write_control(
                cycle / "metrics.json",
                "metrics",
                run.name,
                tasks={},
                longbench={"score": None},
                gsm8k=None,
                math500=None,
                math={"score": None},
                theoretical_storage_pass=False,
                actual_storage_pass=False,
                failures=["no closed execution attempt"],
                baseline_reference={"longbench": {}, "math": {"gsm8k": 0.0, "math500": 2.8}},
                comparison_mode="development_candidate_ranking",
            )
            write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})
            write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name, measurement_valid=False)

            result = decide(config, run, "000")

            self.assertEqual(result["action"], "REFINE")
            self.assertFalse(result["measurement_valid"])
            self.assertIsNone(result["comparisons"]["candidate_gsm8k"])

    def test_formal_commands_use_all_gpus_and_highest_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = root / "run"
            workspace = run / "cycles" / "000" / "workspace"
            workspace.mkdir(parents=True)
            gpus = [{"index": index, "uuid": f"gpu-{index}"} for index in range(8)]
            plan = {
                "evaluation_tier": "formal",
                "timeout_seconds": 43200,
                "workspace": str(workspace),
                "datasets": config.section("evaluation")["longbench_tasks"] + config.section("evaluation")["math_tasks"],
                "execution": {
                    "adapter": "vecinfer", "gpu_count": 8,
                    "batch_size_schedule": [16, 8, 4, 2, 1],
                    "env": {
                        "VECINFER_GENERATION_MAX_LENGTH": "2048",
                        "VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH": "16384",
                    },
                },
            }
            adapter = VecInferAdapter(config)
            with patch.object(adapter, "prepare_datasets", return_value=root / "datasets"):
                commands = adapter.commands(run, "000", "000", gpus, plan)
            self.assertEqual(len(commands), 15)
            self.assertEqual({command.gpu_index for command in commands}, set(range(8)))
            self.assertTrue(all(command.timeout_seconds == 43200 for command in commands))
            self.assertTrue(all(command.argv[command.argv.index("--batch_size") + 1] == "16" for command in commands))
            for gpu in gpus:
                assigned = [command for command in commands if command.gpu_index == gpu["index"]]
                self.assertLessEqual(len(assigned), 2)

    def test_formal_one_row_implementation_smoke_stays_batch_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = root / "run"
            workspace = run / "cycles" / "011" / "workspace"
            workspace.mkdir(parents=True)
            plan = {
                "evaluation_tier": "formal",
                "batch_size": 1,
                "workspace": str(workspace),
                "datasets": ["qasper", "gsm8k"],
                "execution": {
                    "adapter": "vecinfer",
                    "gpu_count": 8,
                    "batch_size_schedule": [16, 8, 4, 2, 1],
                    "env": {
                        "KVRESEARCH_IMPLEMENTATION_SMOKE": "1",
                        "VECINFER_GENERATION_MAX_LENGTH": "512",
                        "VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH": "16384",
                    },
                },
            }
            adapter = VecInferAdapter(config)
            with patch.object(adapter, "prepare_datasets", return_value=root / "smoke-datasets"):
                commands = adapter.commands(
                    run, "011", "smoke-test", [{"index": 0, "uuid": "gpu-0"}], plan,
                    tier_override="formal",
                )

            self.assertEqual(len(commands), 2)
            self.assertTrue(
                all(command.argv[command.argv.index("--batch_size") + 1] == "1" for command in commands)
            )

    def test_30pct_gate_requires_gain_without_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            attempt = root / "attempt"
            tasks = config.section("evaluation")["longbench_tasks"] + config.section("evaluation")["math_tasks"]
            stock = stock_codebook_manifest(config)
            for task in tasks:
                identity = {"task": task, "index": 0}
                atomic_write_jsonl(attempt / "raw" / f"{task}.jsonl", [{
                    "score": 46.0, "sample_idx": 0, "_fixed_sample": identity,
                    "fixed_workload_fingerprint": f"fingerprint-{task}",
                }])
                atomic_write_jsonl(
                    attempt / "arms" / task / "unmodified_vecinfer.jsonl",
                    [{"score": 45.0, "arm_role": "stock_vecinfer_control",
                      "sample_idx": 0, "_fixed_sample": identity,
                      "fixed_workload_fingerprint": f"fingerprint-{task}",
                      "stock_vecinfer_codebook_root": stock["root"],
                      "stock_vecinfer_codebook_manifest_sha256": stock["sha256"],
                      "stock_vecinfer_assignment_policy": "squared_euclidean"}],
                )
            evaluated = [{"score": [True]}]
            with patch("kvresearch.pipeline._math_eval", return_value=(evaluated, {})):
                result = _formal_30pct_gate(config, attempt, tasks)
            self.assertTrue(result["qualified_for_remaining_70pct"])
            for task in config.section("evaluation")["longbench_tasks"]:
                identity = {"task": task, "index": 0}
                atomic_write_jsonl(attempt / "raw" / f"{task}.jsonl", [{
                    "score": 40.0, "sample_idx": 0, "_fixed_sample": identity,
                    "fixed_workload_fingerprint": f"fingerprint-{task}",
                }])
            with patch("kvresearch.pipeline._math_eval", return_value=(evaluated, {})):
                regressed = _formal_30pct_gate(config, attempt, tasks)
            self.assertFalse(regressed["qualified_for_remaining_70pct"])
            self.assertFalse(regressed["no_regressions"])
            qasper = attempt / "raw" / "qasper.jsonl"
            row = read_jsonl(qasper)[0]
            row["fixed_workload_fingerprint"] = "tampered"
            atomic_write_jsonl(qasper, [row])
            with patch("kvresearch.pipeline._math_eval", return_value=(evaluated, {})):
                with self.assertRaisesRegex(RuntimeError, "identity/fingerprint"):
                    _formal_30pct_gate(config, attempt, tasks)

    def test_adapter_rejects_scored_math_cap_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = root / "run"
            workspace = run / "cycles" / "000" / "workspace"
            workspace.mkdir(parents=True)
            plan = {
                "evaluation_tier": "development",
                "workspace": str(workspace),
                "datasets": ["gsm8k"],
                "execution": {
                    "adapter": "vecinfer",
                    "env": {
                        "VECINFER_GENERATION_MAX_LENGTH": "16384",
                        "VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH": "16384",
                    },
                },
            }
            adapter = VecInferAdapter(config)
            with patch.object(adapter, "prepare_datasets", return_value=root / "datasets"):
                with self.assertRaisesRegex(ValueError, "math quality cap drifted"):
                    adapter.commands(run, "000", "000", [{"index": 0, "uuid": "gpu-0"}], plan)

    def test_validator_math_budget_contract_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp))
            plan = {
                "decoding": {
                    "math_max_total_length": 2048,
                    "gsm8k": {"max_total_length": 2048},
                    "math500": {"max_total_length": 2048},
                },
                "execution": {
                    "env": {
                        "VECINFER_GENERATION_MAX_LENGTH": "2048",
                        "VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH": "16384",
                    }
                },
            }
            tasks = {"gsm8k", "math500"}
            self.assertTrue(_math_budget_check(config, plan, tasks)["passed"])
            plan["decoding"]["longbench"] = {"method": "greedy", "max_total_length": 2048}
            self.assertFalse(_math_budget_check(config, plan, tasks)["passed"])
            del plan["decoding"]["longbench"]
            plan["execution"]["env"]["VECINFER_GENERATION_MAX_LENGTH"] = "16384"
            self.assertFalse(_math_budget_check(config, plan, tasks)["passed"])

    def test_same_family_two_small_deltas_trigger_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "cooldown", "cooldown-run")

            def cycle_files(index: int, long_score: float, previous_streak: int | None = None):
                cycle = run / "cycles" / f"{index:03d}"
                atomic_write_jsonl(cycle / "candidates.jsonl", [{"candidate_id": "c", "mechanism_family": "renamed-family-a",
                    "architecture_axis": "kv_bit_allocation", "canonical_mechanism_family": "kv_bit_allocation_architecture"}])
                write_control(cycle / "selection.json", "selection", run.name, selected_candidate_id="c", ranking=[], rejected=[], evidence=[])
                write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, evaluation_tier="development")
                write_control(cycle / "metrics.json", "metrics", run.name, tasks={}, longbench={"score": long_score},
                              gsm8k={"score": 0.0, "correct": 0}, math500={"score": 0.0, "correct": 0}, math={"score": 0.0},
                              theoretical_storage_pass=True, actual_storage_pass=True, failures=[],
                              baseline_reference={"longbench": {}, "math": {"gsm8k": 1.0, "math500": 1.0}})
                write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})
                write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name, measurement_valid=True)
                if previous_streak is not None:
                    write_control(cycle / "decision.json", "decision", run.name, action="REFINE", cooldown={"small_gain_streak": previous_streak})
                return cycle

            cycle_files(0, 10.0, 0)
            cycle_files(1, 10.1)
            first = decide(config, run, "001")
            self.assertEqual(first["cooldown"]["small_gain_streak"], 1)
            cycle_files(2, 10.2)
            second = decide(config, run, "002")
            self.assertTrue(second["cooldown"]["triggered"])
            self.assertEqual(second["action"], "PIVOT")
            self.assertEqual(second["cooldown"]["cooldown_until_cycle"], 4)

    def test_invalid_previous_measurement_does_not_enter_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "technical predecessor", "invalid-previous-run")
            for index in (0, 1):
                cycle = run / "cycles" / f"{index:03d}"
                atomic_write_jsonl(
                    cycle / "candidates.jsonl",
                    [{"candidate_id": "c", "mechanism_family": f"alias-{index}",
                      "architecture_axis": "kv_bit_allocation", "canonical_mechanism_family": "kv_bit_allocation_architecture"}],
                )
                write_control(
                    cycle / "selection.json", "selection", run.name,
                    selected_candidate_id="c", ranking=[], rejected=[], evidence=[],
                )
                write_control(
                    cycle / "experiment_plan.json", "experiment_plan", run.name,
                    evaluation_tier="development",
                )
            write_control(
                run / "cycles" / "000" / "metrics.json", "metrics", run.name,
                status="failed", failures=["missing predictions"],
                comparison_mode="development_candidate_ranking",
                longbench={"score": None}, gsm8k={"correct": 0}, math500={"correct": 0},
                theoretical_storage_pass=False, actual_storage_pass=False,
            )
            write_control(
                run / "cycles" / "000" / "experiment_diagnosis.json",
                "experiment_diagnosis", run.name, measurement_valid=False,
            )
            write_control(
                run / "cycles" / "000" / "decision.json", "decision", run.name,
                action="REFINE", cooldown={"small_gain_streak": 0},
            )
            write_control(
                run / "cycles" / "001" / "metrics.json", "metrics", run.name,
                status="complete", failures=[], comparison_mode="development_candidate_ranking",
                tasks={}, longbench={"score": 10.0},
                gsm8k={"score": 0.0, "correct": 0},
                math500={"score": 0.0, "correct": 0}, math={"score": 0.0},
                theoretical_storage_pass=True, actual_storage_pass=True,
                baseline_reference={"longbench": {}, "math": {"gsm8k": 0.0, "math500": 0.0}},
            )
            write_control(
                run / "cycles" / "001" / "verified_registry.json",
                "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={},
            )
            write_control(
                run / "cycles" / "001" / "experiment_diagnosis.json",
                "experiment_diagnosis", run.name, measurement_valid=True,
            )

            result = decide(config, run, "001")

            self.assertEqual(result["cooldown"]["small_gain_streak"], 0)
            self.assertFalse(result["cooldown"]["triggered"])

    def test_invalid_cycle_does_not_exhaust_three_valid_attempt_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "valid research window", "valid-window-run")

            def cycle_files(index: int, *, valid: bool):
                cycle = run / "cycles" / f"{index:03d}"
                write_control(
                    cycle / "experiment_plan.json", "experiment_plan", run.name,
                    evaluation_tier="development",
                )
                write_control(
                    cycle / "metrics.json", "metrics", run.name,
                    status="complete" if valid else "failed",
                    failures=[] if valid else ["technical failure"],
                    tasks={"qasper": {"score": 10.0}},
                    comparison_mode="development_candidate_ranking",
                    longbench={"score": 10.0},
                    gsm8k={"score": 0.0, "correct": 0},
                    math500={"score": 0.0, "correct": 0},
                    math={"score": 0.0},
                    theoretical_storage_pass=valid,
                    actual_storage_pass=valid,
                    baseline_reference={"longbench": {"qasper": 11.0}, "math": {"gsm8k": 1.0, "math500": 1.0}},
                )
                write_control(
                    cycle / "verified_registry.json", "verified_registry", run.name,
                    values=[], rejected_claims=[], source_hashes={},
                )
                write_control(
                    cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name,
                    measurement_valid=valid,
                )
                return cycle

            cycle_files(0, valid=False)
            write_control(
                run / "cycles" / "000" / "decision.json", "decision", run.name,
                action="REFINE", cooldown={"small_gain_streak": 0}, measurement_valid=False,
            )
            cycle_files(1, valid=True)
            write_control(
                run / "cycles" / "001" / "decision.json", "decision", run.name,
                action="REFINE", cooldown={"small_gain_streak": 0}, measurement_valid=True,
            )
            cycle_files(2, valid=True)

            result = decide(config, run, "002")

            self.assertEqual(result["action"], "REFINE")
            self.assertEqual(result["comparisons"]["valid_development_attempts_in_window"], 2)

    def test_three_valid_attempts_trigger_research_pivot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "three valid attempts", "three-valid-run")
            for index in range(3):
                cycle = run / "cycles" / f"{index:03d}"
                write_control(cycle / "experiment_plan.json", "experiment_plan", run.name, evaluation_tier="development")
                write_control(
                    cycle / "metrics.json", "metrics", run.name,
                    tasks={"qasper": {"score": 10.0}}, failures=[],
                    comparison_mode="development_candidate_ranking",
                    longbench={"score": 10.0}, gsm8k={"score": 0.0, "correct": 0},
                    math500={"score": 0.0, "correct": 0}, math={"score": 0.0},
                    theoretical_storage_pass=True, actual_storage_pass=True,
                    baseline_reference={"longbench": {"qasper": 11.0}, "math": {"gsm8k": 1.0, "math500": 1.0}},
                )
                write_control(cycle / "verified_registry.json", "verified_registry", run.name, values=[], rejected_claims=[], source_hashes={})
                write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name, measurement_valid=True)
                if index < 2:
                    write_control(cycle / "decision.json", "decision", run.name, action="REFINE", cooldown={"small_gain_streak": 0})

            result = decide(config, run, "002")

            self.assertEqual(result["action"], "PIVOT")
            self.assertEqual(result["comparisons"]["valid_development_attempts_in_window"], 3)

    def test_recomputed_lesson_replaces_stale_memory_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = test_config(root)
            run = initialize_run(config, "lesson upsert", "lesson-upsert-run")
            cycle = run / "cycles" / "000"
            write_control(
                cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name,
                deficiencies=[{
                    "type": "none", "severity": "info", "evidence": ["metrics.json"],
                    "suggested_action": "evaluate quality thresholds",
                }],
            )
            write_control(cycle / "decision.json", "decision", run.name, action="PIVOT", cooldown={})
            learn(run, "000")
            write_control(cycle / "decision.json", "decision", run.name, action="REFINE", cooldown={})
            learn(run, "000")

            memory_rows = read_jsonl(config.project_root / "memory" / "lessons.jsonl")
            matching = [row for row in memory_rows if row.get("lesson_id") == "lesson-upsert-run-000-000"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["decision"], "REFINE")


if __name__ == "__main__":
    unittest.main()
