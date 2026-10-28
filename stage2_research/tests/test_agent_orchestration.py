from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kvresearch.agent import (
    STAGE_WORKERS,
    WORKER_AGENT_RUNTIME,
    _is_transient_agent_failure,
    _run_agent_command,
    create_task_packet,
    run_agent_action,
    validate_worker_artifacts,
)
from kvresearch.codex_backend import main as codex_main
from kvresearch.initialize import initialize_run
from kvresearch.navigator import Action, Navigator
from kvresearch.schema import ArtifactValidator, control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref, read_json, sha256_file
from tests.helpers import architecture_candidates, test_config, write_control


class AgentOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = test_config(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_stage_worker_topology(self):
        self.assertEqual({step: len(workers) for step, workers in STAGE_WORKERS.items()}, {
            "literature": 4, "evidence_synthesis": 3, "problem_diagnosis": 3, "candidates": 4, "selection": 3,
        })

    def test_task_packet_is_stable_and_workers_are_a_hard_gate(self):
        run = initialize_run(self.config, "parallel evidence", "parallel-run")
        action = Navigator(self.config).next(run)
        packet_path, prompt_path = create_task_packet(self.config, run, action)
        second_packet, second_prompt = create_task_packet(self.config, run, action)
        self.assertEqual((packet_path, prompt_path), (second_packet, second_prompt))
        packet = read_json(packet_path)
        self.assertTrue(packet["parallel"])
        self.assertEqual(len(packet["workers"]), 4)
        self.assertIn("spawn all assigned subagents concurrently", prompt_path.read_text(encoding="utf-8"))
        self.assertIn("fork_turns='none'", prompt_path.read_text(encoding="utf-8"))
        self.assertIn("model='gpt-6.1-sol'", prompt_path.read_text(encoding="utf-8"))
        self.assertIn("reasoning_effort='xhigh'", prompt_path.read_text(encoding="utf-8"))
        self.assertIn("service_tier='fast'", prompt_path.read_text(encoding="utf-8"))

        for assignment in packet["workers"][:-1]:
            atomic_write_json(Path(assignment["output"]), {
                "worker_id": assignment["worker_id"], "role": assignment["role"], "status": "complete",
                "agent_runtime": WORKER_AGENT_RUNTIME,
                "findings": [], "evidence": [], "recommendation": "done",
            })
        valid, errors, manifest = validate_worker_artifacts(packet_path)
        self.assertFalse(valid)
        self.assertTrue(errors)
        self.assertIsNone(manifest)

        assignment = packet["workers"][-1]
        atomic_write_json(Path(assignment["output"]), {
            "worker_id": assignment["worker_id"], "role": assignment["role"], "status": "complete",
            "agent_runtime": WORKER_AGENT_RUNTIME,
            "findings": [], "evidence": [], "recommendation": "done",
        })
        valid, errors, manifest = validate_worker_artifacts(packet_path)
        self.assertTrue(valid, errors)
        self.assertTrue(manifest and manifest.is_file())
        self.assertEqual(read_json(manifest)["worker_count"], 4)

        first_worker = Path(packet["workers"][0]["output"])
        original_worker = read_json(first_worker)
        for field, invalid_value in (("model", "wrong-model"), ("reasoning_effort", "high"),
                                     ("service_tier", "default"), ("service_tier", None)):
            with self.subTest(field=field, invalid_value=invalid_value):
                invalid_worker = json.loads(json.dumps(original_worker))
                if invalid_value is None:
                    invalid_worker["agent_runtime"].pop(field)
                else:
                    invalid_worker["agent_runtime"][field] = invalid_value
                atomic_write_json(first_worker, invalid_worker)
                valid, errors, manifest = validate_worker_artifacts(packet_path)
                self.assertFalse(valid)
                self.assertTrue(any("agent_runtime" in error for error in errors))
                self.assertIsNone(manifest)

    def test_backend_applies_requested_runtime_to_parent_and_subagent_defaults(self):
        prompt_path = self.root / "prompt.md"
        prompt_path.write_text("Run the assigned stage.", encoding="utf-8")
        with patch("kvresearch.codex_backend.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = codex_main(["--prompt", str(prompt_path), "--run", str(self.root)])
        self.assertEqual(result, 0)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "gpt-6.1-sol")
        overrides = [command[index + 1] for index, item in enumerate(command) if item == "-c"]
        self.assertIn('model_reasoning_effort="xhigh"', overrides)
        self.assertIn('service_tier="fast"', overrides)
        self.assertIn('features.fast_mode=true', overrides)
        self.assertIn('agents.default_subagent_model="gpt-6.1-sol"', overrides)
        self.assertIn('agents.default_subagent_reasoning_effort="xhigh"', overrides)

    def test_runtime_change_invalidates_cached_worker_task(self):
        run = initialize_run(self.config, "runtime-bound evidence", "runtime-bound-run")
        action = Navigator(self.config).next(run)
        original_packet, _ = create_task_packet(self.config, run, action)
        with patch.dict(WORKER_AGENT_RUNTIME, {"service_tier": "default"}):
            changed_packet, _ = create_task_packet(self.config, run, action)
        self.assertNotEqual(original_packet, changed_packet)

    def test_candidate_semantics_require_four_worker_lanes(self):
        path = self.root / "candidates.jsonl"
        rows = architecture_candidates()
        atomic_write_jsonl(path, rows)
        validator = ArtifactValidator(self.config.catalog_path)
        self.assertTrue(validator.validate_jsonl(path, "candidates").valid)
        atomic_write_jsonl(path, rows[:3])
        self.assertFalse(validator.validate_jsonl(path, "candidates").valid)

    def test_agent_family_label_cannot_change_host_classification(self):
        rows = architecture_candidates()
        rows[0]["mechanism_family"] = "totally-new-name"
        rows[0]["canonical_mechanism_family"] = "totally-new-name"
        path = self.root / "renamed.jsonl"
        atomic_write_jsonl(path, rows)
        result = ArtifactValidator(self.config.catalog_path).validate_jsonl(path, "candidates")
        self.assertFalse(result.valid)
        self.assertTrue(any("E_FAMILY_CLASSIFICATION" in error for error in result.errors))

    def test_agent_capacity_failure_retries_then_succeeds(self):
        capacity = subprocess.CompletedProcess(["agent"], 1, "", "Selected model is at capacity. Please try a different model.")
        success = subprocess.CompletedProcess(["agent"], 0, "done", "")
        with patch("kvresearch.agent.subprocess.run", side_effect=[capacity, capacity, success]) as run, \
             patch("kvresearch.agent.time.sleep") as sleep:
            result, attempts = _run_agent_command(
                ["agent"], self.root, transient_retries=4,
                initial_backoff_seconds=2, max_backoff_seconds=8,
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_non_transient_agent_failure_is_not_retried(self):
        invalid = subprocess.CompletedProcess(["agent"], 1, "", "schema validation failed")
        self.assertFalse(_is_transient_agent_failure(invalid))
        with patch("kvresearch.agent.subprocess.run", return_value=invalid) as run, \
             patch("kvresearch.agent.time.sleep") as sleep:
            result, attempts = _run_agent_command(
                ["agent"], self.root, transient_retries=6,
                initial_backoff_seconds=2, max_backoff_seconds=8,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(attempts), 1)
        run.assert_called_once()
        sleep.assert_not_called()

    def test_execution_repair_preserves_host_snapshot_and_never_self_references(self):
        run = initialize_run(self.config, "repair manifest", "manifest-repair-run")
        cycle = run / "cycles" / "000"
        selection = cycle / "selection.json"
        plan = cycle / "experiment_plan.json"
        write_control(selection, "selection", run.name,
                      selected_candidate_id="c1", ranking=["c1", "c2", "c3", "c4"], rejected=[], evidence=[])
        write_control(plan, "experiment_plan", run.name, candidate_id="c1", code_scope=["x.py"],
                      evaluation_tier="development", datasets=[], decoding={"math_max_total_length": 2048},
                      outputs=[], budgets={}, decision_rules={}, storage_accounting={},
                      workspace=str(cycle / "workspace"), execution={})
        output = cycle / "implementation_manifest.json"
        atomic_write_json(output, control_envelope(
            "implementation_manifest", run.name, "kvresearch.workspace",
            [path_ref(selection), path_ref(plan)], candidate_id="c1",
            workspace=str(cycle / "workspace"), base_hash="a", changed_files=["x.py"],
            patch_hash="b", build_command=[], run_entrypoint="scripts.modeldb.main_pq", dependencies=[],
        ))
        before = sha256_file(output)
        action = Action("READY", "execution_repair", "agent", (str(plan), str(output)), str(output))
        success = subprocess.CompletedProcess(["agent"], 0, "done", "")
        with patch("kvresearch.agent._run_agent_command", return_value=(success, [success])):
            result = run_agent_action(self.config, run, action)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(sha256_file(output), before)
        self.assertFalse(any(ref["path"] == str(output) for ref in read_json(output)["inputs"]))


if __name__ == "__main__":
    unittest.main()
