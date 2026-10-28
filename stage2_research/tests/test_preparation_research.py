"""Default-engine contract tests; all agent output and scores are synthetic."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kvresearch.config import Config, load_config
from kvresearch import research
from kvresearch.research_agent import RUNTIME, run_agent, snapshot_fields, validate_result
from kvresearch.research_data import checked_control, freeze_datasets, validate_context
from kvresearch.research_execution import assess, execute, phase_sources, read_predictions
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref, read_json, read_jsonl, sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stage1_preparation import create_try
from stage3_output import export_method
import workflow


EVALUATOR = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("arm", "dataset", "output", "task_id"):
    p.add_argument("--" + name, required=True)
a = p.parse_args()
rows = [json.loads(line) for line in Path(a.dataset).read_text().splitlines()]
score = 0.7 if a.arm == "candidate" else 0.5
with open(a.output, "w") as out:
    for row in rows:
        out.write(json.dumps({"sample_id": row["sample_id"], "status": "complete",
            "metrics": {"quality": score, "loss": 1 - score}, "constraints": {"cost": 0.8}}) + "\\n")
'''


class PreparationResearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw.jsonl"
        atomic_write_jsonl(self.source, [{"question": str(i), "answer": i} for i in range(100)])
        values = {"goal_baseline": "提高质量；比较本次指定的参考算法", "model": "本次指定的测试模型",
                  "datasets": str(self.source), "hardware": "CPU，只用于框架测试"}
        self.case = create_try(self.root, values)
        data = copy.deepcopy(load_config().data)
        data["project"].update(root=str(self.root), runs_dir=str(self.case / "runs"))
        data["automation"]["agent_command"] = ["fake-agent", "{prompt}", "{run}"]
        path = self.case / "config.kvresearch.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.config = Config(path, data)
        self.run = research.initialize(self.config, (self.case / "preparation.md").read_text(),
                                       self.case.name, self.case / "preparation.md")
        self.context = {
            "ready": True, "goal": "提高本次任务质量", "model": {"name": "UserChosenModel"},
            "baseline": {"name": "UserChosenBaseline"},
            "datasets": [{"id": "user_task", "source": {"format": "jsonl", "path": str(self.source)}}],
            "hardware": {"mode": "local", "gpu_ids": [], "python": sys.executable},
            "metrics": [{"id": "quality", "label": "用户质量指标", "tasks": ["user_task"], "direction": "maximize"}],
            "evaluation_protocol": {"scorer": "synthetic test only", "seed": 42},
            "constraints": [],
        }
        self.steps = []

    def seed_context(self):
        return research._artifact(self.run, self.run / "research_context.json", "research_context",
                                  [self.case / "preparation.md"], **self.context)

    def proposals(self):
        return [{"candidate_id": f"proposal-{i}", "worker_id": f"worker-{i:02d}",
                 "candidate_kind": "research_proposal", "mechanism_family": f"family-{i}",
                 "mechanism": f"independent idea {i}", "causal_chain": ["cause", "effect"],
                 "budget_design": "user-defined resources", "minimal_code_change": "evaluator.py", "falsification": "no gain"}
                for i in range(3)]

    def fake_backend(self, argv, cwd, **kwargs):
        task = read_json(Path(argv[1]).parent / "task.json")
        output = Path(task["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        step = task["step"]
        self.steps.append(step)
        for assignment in task["workers"]:
            atomic_write_json(Path(assignment["output"]), {"worker_id": assignment["worker_id"],
                "role": assignment["role"], "status": "complete", "agent_runtime": RUNTIME,
                "findings": ["synthetic fixture"], "evidence": [], "recommendation": "synthetic fixture"})
        if step == "research_intake":
            value = self.context
        elif step == "literature":
            atomic_write_jsonl(output, [{"paper_id": "fixture", "title": "test evidence", "source_url": "https://example.org", "relevance": "test"}])
            value = None
        elif step == "candidates":
            atomic_write_jsonl(output, self.proposals())
            value = None
        elif step == "selection":
            value = {"selected_candidate_id": "proposal-0", "ranking": [f"proposal-{i}" for i in range(3)], "rejected": [], "evidence": []}
        elif step == "experiment_plan":
            formal = any(Path(ref["path"]).name == "decision.json" and read_json(Path(ref["path"])).get("action") == "PROMOTE" for ref in task["inputs"])
            value = {"candidate_id": "proposal-0", "evaluation_tier": "formal" if formal else "development",
                     "datasets": ["user_task"], "workspace": str(output.parent / "workspace"),
                     "execution": {"command": [sys.executable, "evaluator.py", "--arm", "{arm}",
                        "--dataset", "{dataset}", "--output", "{output}", "--task_id", "{task_id}"], "timeout_seconds": 10},
                     "code_scope": ["evaluator.py"], "decision_rules": "quality gain"}
        elif step in {"implementation_manifest", "implementation_repair", "execution_repair"}:
            workspace = output.parent / "workspace"
            (workspace / "evaluator.py").write_text(EVALUATOR)
            value = {"workspace": str(workspace), "implementation_notes": "synthetic evaluator, never real inference"}
        else:
            from kvresearch.research_agent import REQUIRED
            value = {key: ["synthetic evidence"] for key in REQUIRED[step]}
        if value is not None:
            atomic_write_json(output, value)
        result = subprocess.CompletedProcess(argv, 0, "", "")
        return result, [result]

    def advance_to(self, step):
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend):
            for _ in range(50):
                action = research.next_action(self.config, self.run)
                if action.step == step:
                    return action
                research.perform(self.config, self.run, action)
        self.fail(f"did not reach {step}")

    def test_default_requires_only_preparation_before_intake(self):
        for name in ("model", "baseline", "evaluation", "remote", "storage"):
            self.assertNotIn(name, self.config.data)
        first = research.next_action(self.config, self.run)
        self.assertEqual(first.step, "research_intake")
        self.assertIn(str(self.case / "preparation.md"), first.inputs)
        self.assertFalse((self.run / "datasets.lock.json").exists())

    def test_missing_user_material_is_needs_input(self):
        self.context = {"ready": False, "missing_inputs": ["用户未指定基线的可执行入口"]}
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend), contextlib.redirect_stdout(io.StringIO()):
            result = research.run_loop(self.config, self.run, 4, self.case / "reports")
        self.assertEqual(result["state"], "NEEDS_INPUT")
        self.assertFalse((self.run / "datasets.lock.json").exists())

    def test_fixed_total_5_30_65_and_resume_identity(self):
        self.seed_context()
        lock = freeze_datasets(self.config, self.run)
        entry = lock["tasks"]["user_task"]
        self.assertEqual([entry["partitions"][p]["rows"] for p in ("development", "formal", "remaining")], [5, 30, 65])
        groups = [set(entry["partitions"][p]["sample_ids"]) for p in ("development", "formal", "remaining")]
        self.assertEqual(len(set.union(*groups)), 100)
        self.assertEqual(sum(map(len, groups)), 100)
        before = sha256_file(self.run / "datasets.lock.json")
        self.assertEqual(freeze_datasets(self.config, self.run), lock)
        self.assertEqual(before, sha256_file(self.run / "datasets.lock.json"))
        self.assertIn("| user_task | 100 | 5 | 30 | 65 |", (self.case / "dataset_split.md").read_text())

    def test_partition_or_source_tampering_is_rejected(self):
        self.seed_context()
        lock = freeze_datasets(self.config, self.run)
        path = Path(lock["tasks"]["user_task"]["partitions"]["formal"]["path"])
        original = path.read_bytes()
        path.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "变化"):
            freeze_datasets(self.config, self.run)
        path.write_bytes(original)
        self.source.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "变化"):
            freeze_datasets(self.config, self.run)

    def test_duplicates_and_small_counts_stay_disjoint(self):
        atomic_write_jsonl(self.source, [{"question": "same"}] * 21)
        self.seed_context()
        entry = freeze_datasets(self.config, self.run)["tasks"]["user_task"]
        self.assertEqual([entry["partitions"][p]["rows"] for p in ("development", "formal", "remaining")], [2, 7, 12])
        self.assertEqual(len({row["sample_id"] for row in read_jsonl(Path(entry["full"]["path"]))}), 21)

    def test_other_fraction_cannot_silently_change_protocol(self):
        self.seed_context()
        self.config.data["dataset_split"]["formal_fraction"] = .285
        with self.assertRaisesRegex(ValueError, "占总量30%"):
            freeze_datasets(self.config, self.run)

    def test_missing_required_resource_is_task_specific(self):
        self.context["resources"] = [{"role": "用户指定的适配器", "path": str(self.root / "absent"), "required": True}]
        self.seed_context()
        with self.assertRaisesRegex(FileNotFoundError, "用户指定的适配器"):
            freeze_datasets(self.config, self.run)

    def test_three_selectable_candidates_without_comparator(self):
        proposals = self.proposals()
        validate_result("candidates", proposals, [])
        with self.assertRaisesRegex(ValueError, "三个"):
            validate_result("candidates", proposals + [proposals[0]], [])
        proposals[2]["candidate_kind"] = "stock_control"
        with self.assertRaisesRegex(ValueError, "科研方案"):
            validate_result("candidates", proposals, [])

    def test_selection_excludes_active_cooldown_family(self):
        cycle = self.run / "cycles" / "003"
        candidates = cycle / "candidates.jsonl"
        atomic_write_jsonl(candidates, self.proposals())
        decision = self.run / "cycles" / "002" / "decision.json"
        research._artifact(self.run, decision, "decision", [], cooldown={"active_families": [{"mechanism_family": "family-0", "until_cycle": 4}]})
        value = {"selected_candidate_id": "proposal-0", "ranking": [f"proposal-{i}" for i in range(3)], "rejected": [], "evidence": []}
        with self.assertRaisesRegex(ValueError, "冷却"):
            validate_result("selection", value, [candidates, decision])

    def test_full_flow_promotes_then_runs_30_65_5_and_exports(self):
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend), contextlib.redirect_stdout(io.StringIO()):
            result = research.run_loop(self.config, self.run, 100, self.case / "reports")
        self.assertEqual(result["state"], "DONE", result)
        first, final = [self.run / "cycles" / name for name in ("000", "001")]
        self.assertEqual(read_json(first / "decision.json")["action"], "PROMOTE")
        self.assertEqual(read_json(final / "decision.json")["action"], "STOP")
        self.assertFalse((final / "selection.json").exists())
        phases = read_json(final / "attempts" / "000" / "exit.json")["phases"]
        self.assertEqual(phases, ["formal", "remaining", "development"])
        self.assertEqual(sum(len(read_jsonl(final / "attempts" / "000" / p / "candidate" / "user_task.jsonl")) for p in phases), 100)
        self.assertEqual(self.steps.count("candidates"), 1)
        metrics = read_json(final / "metrics.json")
        self.assertTrue(metrics["coverage"]["full_dataset"])
        report = Path(result["final_method"]).read_text()
        self.assertIn("用户质量指标", report)
        self.assertIn("full_5_30_65", report)
        self.assertNotIn("LongBench", report)
        self.assertEqual(export_method(self.case, self.run), Path(result["final_method"]))

    def test_final_gate_failure_does_not_read_remaining(self):
        action = self.advance_to("execute")
        cycle = Path(action.output).parents[2]
        plan = read_json(cycle / "experiment_plan.json")
        plan["evaluation_tier"] = "formal"
        atomic_write_json(cycle / "experiment_plan.json", plan)
        # Deliberately equal arm scores, so the formal gate cannot pass.
        (cycle / "workspace" / "evaluator.py").write_text(EVALUATOR.replace('0.7 if a.arm == "candidate" else 0.5', '0.5'))
        result = execute(self.config, self.run, cycle, Path(action.output).parent)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["phases"], ["formal"])
        self.assertFalse(result["qualified_for_completion"])
        self.assertFalse((Path(action.output).parent / "remaining").exists())

    def test_predicted_ids_and_finite_measurements_are_required(self):
        self.seed_context()
        lock = freeze_datasets(self.config, self.run)
        source = phase_sources(lock, "development")["user_task"]
        output = self.root / "predictions.jsonl"
        atomic_write_jsonl(output, [{"sample_id": "wrong", "status": "complete", "metrics": {"quality": 1}}])
        with self.assertRaisesRegex(ValueError, "身份"):
            read_predictions(self.context, source, output, "user_task")
        rows = [{"sample_id": row["sample_id"], "status": "complete", "metrics": {"quality": float("nan")}} for row in read_jsonl(source)]
        atomic_write_jsonl(output, rows)
        with self.assertRaisesRegex(ValueError, "非有限"):
            read_predictions(self.context, source, output, "user_task")

    def test_minimize_metric_and_budget_constraints(self):
        self.context["metrics"] = [{"id": "loss", "tasks": ["user_task"], "direction": "minimize"}]
        self.context["constraints"] = [{"id": "cost", "operator": "<=", "limit": .7}]
        self.seed_context()
        lock = freeze_datasets(self.config, self.run)
        source = phase_sources(lock, "development")["user_task"]
        directory = self.root / "development"
        for arm, loss in (("baseline", .5), ("candidate", .3)):
            atomic_write_jsonl(directory / arm / "user_task.jsonl", [{"sample_id": row["sample_id"], "status": "complete", "metrics": {"loss": loss}, "constraints": {"cost": .8}} for row in read_jsonl(source)])
        result = assess(self.context, [directory], {"development": {"user_task": source}})
        self.assertTrue(result["any_strict_gain"])
        self.assertFalse(result["constraints_pass"])
        self.assertEqual(len(result["constraint_violations"]), 5)

    def test_untracked_code_invalidates_implementation(self):
        action = self.advance_to("validate-cycle")
        cycle = Path(action.output).parent
        manifest = cycle / "implementation_manifest.json"
        self.assertTrue(research._implementation_current(manifest))
        (cycle / "workspace" / "extra.py").write_text("x = 1\n")
        self.assertFalse(research._implementation_current(manifest))

    def test_build_products_are_bound_before_smoke(self):
        action = self.advance_to("validate-cycle")
        cycle = Path(action.output).parent
        manifest = cycle / "implementation_manifest.json"
        value = read_json(manifest)
        value["build_commands"] = [[sys.executable, "-c", "from pathlib import Path; Path('built.bin').write_bytes(b'fixture')"]]
        atomic_write_json(manifest, value)
        result = research.validate_cycle(self.config, self.run, cycle)
        self.assertTrue(result["all_hard_checks_pass"], result)
        self.assertIn("built.bin", read_json(manifest)["file_hashes"])
        self.assertTrue(research._implementation_current(manifest))

    def test_changed_preparation_prevents_resume(self):
        (self.case / "preparation.md").write_text("changed")
        self.assertEqual(research.next_action(self.config, self.run).state, "INVALID")

    def test_unchanged_source_repair_still_creates_a_fresh_attempt(self):
        action = self.advance_to("execute")
        attempt = Path(action.output).parent
        cycle = attempt.parents[1]
        research.perform(self.config, self.run, action)
        exit_data = read_json(attempt / "exit.json")
        exit_data.update(status="failed", error="synthetic transient environment failure")
        atomic_write_json(attempt / "exit.json", exit_data)
        repair = research.next_action(self.config, self.run)
        self.assertEqual(repair.step, "execution_repair")
        before = read_json(cycle / "implementation_manifest.json")
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend):
            result = research.perform(self.config, self.run, repair)
        self.assertEqual(result["status"], "complete", result)
        after = read_json(cycle / "implementation_manifest.json")
        self.assertEqual(before["file_hashes"], after["file_hashes"])
        self.assertGreater(after["revision"], before["revision"])
        next_execute = self.advance_to("execute")
        self.assertEqual(Path(next_execute.output).parent.name, "001")
        self.assertEqual(Path(next_execute.output).parents[2], cycle)
        self.assertFalse((cycle / "decision.json").exists())

    def test_resource_wait_requires_explicit_resume(self):
        action = self.advance_to("validate-cycle")
        with patch("kvresearch.research.preflight", return_value={"available": False, "errors": ["busy"]}):
            result = research.perform(self.config, self.run, action)
        self.assertTrue(result["resource_wait"])
        self.assertEqual(research.next_action(self.config, self.run).state, "WAIT")
        with patch.dict("os.environ", {"KVRESEARCH_RESUME_REMOTE_ONCE": "1"}):
            self.assertEqual(research.next_action(self.config, self.run).step, "validate-cycle")

    def test_budget_terminal_exports_actual_candidate_without_claiming_success(self):
        self.config.data["automation"]["max_cycles"] = 1
        # Re-lock the on-disk configuration before any scientific work.
        self.config.path.write_text(yaml.safe_dump(self.config.data))
        for name in ("run_manifest.json", "inputs.lock.json"):
            value = read_json(self.run / name)
            for ref in value["inputs"]:
                if ref["path"] == str(self.config.path):
                    ref["sha256"] = sha256_file(self.config.path)
            atomic_write_json(self.run / name, value)
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend), contextlib.redirect_stdout(io.StringIO()):
            result = research.run_loop(self.config, self.run, 100, self.case / "reports")
        self.assertEqual(result["state"], "DONE", result)
        self.assertEqual(read_json(self.run / "terminal.json")["termination_type"], "cycle_budget_exhausted")
        self.assertIn("未记录目标达成", Path(result["final_method"]).read_text())

    def test_workflow_default_reaches_intake_without_legacy_assets(self):
        # A distinct, empty try tests the public entrypoint rather than initialize.
        case = create_try(self.root, {"goal_baseline": "goal", "model": "model", "datasets": str(self.source), "hardware": "CPU"})
        with patch("kvresearch.research_agent._run_agent_command", side_effect=self.fake_backend), contextlib.redirect_stdout(io.StringIO()):
            result = workflow.run_research(case, self.config.path, max_steps=1)
        self.assertEqual(result["step"], "freeze-data")
        self.assertEqual(read_json(Path(result["run_dir"]) / "research_context.json")["model"]["name"], "UserChosenModel")


if __name__ == "__main__":
    unittest.main()
