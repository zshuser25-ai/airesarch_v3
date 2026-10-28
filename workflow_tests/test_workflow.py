from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import workflow
from kvresearch.navigator import Action
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref, read_json
from stage1_preparation import collect_inputs, create_try, resolve_try
from stage3_output import export_method
from tests.helpers import test_config


VALUES = {
    "goal_baseline": "相同 KV 预算下提升精度\n对比基线：stock VecInfer",
    "model": "Llama 3.1 8B Instruct，模型地址 /models/llama",
    "datasets": "LongBench、GSM8K 和 MATH500\n地址 /data/evaluation",
    "hardware": "8 张 GPU，单卡 24GB，节点 gpu-node",
}


class PreparationTests(unittest.TestCase):
    def test_start_command_creates_preparation_without_starting_research(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.StringIO()
            with patch.object(workflow, "PROJECT_ROOT", root), \
                 patch.object(workflow, "run_research") as research, \
                 contextlib.redirect_stdout(output):
                code = workflow.main(["start", "--goal-baseline", VALUES["goal_baseline"],
                                      "--model", VALUES["model"], "--datasets", VALUES["datasets"],
                                      "--hardware", VALUES["hardware"]])
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["stage"], "preparation_complete")
            self.assertTrue((root / "try001" / "preparation.md").is_file())
            research.assert_not_called()

    def test_prompts_for_all_fields_and_retries_empty_answer(self):
        answers = iter([".", "目标与基线", ".", "模型", ".", "数据集", ".", "硬件", "."])
        messages = []
        values = collect_inputs({}, input_fn=lambda _: next(answers), output_fn=messages.append)
        self.assertEqual(values, {"goal_baseline": "目标与基线", "model": "模型", "datasets": "数据集", "hardware": "硬件"})
        self.assertTrue(any("不能为空" in message for message in messages))

    def test_multiline_inputs_survive_markdown_and_new_tries_preserve_old_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_try(root, VALUES)
            original = (first / "preparation.md").read_bytes()
            second = create_try(root, VALUES)
            self.assertEqual((first.name, second.name), ("try001", "try002"))
            self.assertEqual((first / "preparation.md").read_bytes(), original)
            for value in VALUES.values():
                self.assertIn(value, original.decode())
            self.assertFalse((first / "runs").exists())
            self.assertFalse((first / "final_method.md").exists())

    def test_incomplete_inputs_or_interrupted_prompt_do_not_create_try(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "可用硬件"):
                create_try(root, {key: value for key, value in VALUES.items() if key != "hardware"})
            with self.assertRaisesRegex(ValueError, "尚未填写完成"):
                collect_inputs({}, input_fn=lambda _: (_ for _ in ()).throw(EOFError()), output_fn=lambda _: None)
            self.assertEqual(list(root.iterdir()), [])

    def test_concurrent_starts_allocate_distinct_numbers_and_skip_reserved_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "try001").mkdir()
            (root / "try003").write_text("reserved")
            with ThreadPoolExecutor(max_workers=4) as pool:
                directories = list(pool.map(lambda _: create_try(root, VALUES), range(4)))
            self.assertEqual({path.name for path in directories}, {"try004", "try005", "try006", "try007"})

    def test_try_resolution_rejects_outside_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            created = create_try(root, VALUES)
            self.assertEqual(resolve_try(root, created.name), created)
            with self.assertRaisesRegex(ValueError, "项目根目录"):
                resolve_try(root, "../try001")


class ResearchHandoffTests(unittest.TestCase):
    def test_every_agent_receives_original_markdown_and_run_stays_inside_try(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            case = create_try(root, VALUES)
            expected_run = case / "runs" / case.name
            action = Action("READY", "literature", "agent", (str(expected_run / "request.md"),),
                            str(expected_run / "evidence" / "literature.jsonl"))
            waiting = Action("WAIT", "remote_resources", "terminal", (), None)
            with patch("kvresearch.navigator.Navigator") as navigator, \
                 patch("kvresearch.agent.run_agent_action", return_value={"status": "complete"}) as agent, \
                 contextlib.redirect_stdout(io.StringIO()):
                navigator.return_value.next.side_effect = [action, waiting]
                result = workflow.run_research(case, config.path, max_steps=1)
            received = agent.call_args.args[2]
            self.assertEqual(received.inputs[0], str(case / "preparation.md"))
            self.assertEqual((expected_run / "request.md").read_bytes(), (case / "preparation.md").read_bytes())
            self.assertEqual(result["state"], "WAIT")
            self.assertNotIn("final_method", result)
            self.assertEqual(agent.call_args.args[0].runs_dir, case / "runs")
            self.assertTrue((case / "reports" / "kvresearch_stage_report.md").is_file())
            self.assertFalse((case / "final_method.md").exists())

            (case / "preparation.md").write_text("用户修改了已锁定的研究定义\n")
            with self.assertRaisesRegex(ValueError, "锁定请求"):
                workflow.run_research(case, config.path, max_steps=1)

    def test_missing_external_materials_do_not_leave_a_partial_research_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            case = create_try(root, VALUES)
            Path(config.section("project")["formal_baseline"]).unlink()
            with self.assertRaises(FileNotFoundError):
                workflow.run_research(case, config.path, max_steps=1)
            self.assertFalse((case / "runs").exists())
            self.assertTrue((case / "preparation.md").is_file())


class OutputTests(unittest.TestCase):
    def _completed_run(self, root: Path, *, achieved: bool = True):
        case = create_try(root, VALUES)
        run = case / "runs" / case.name
        run.mkdir(parents=True)
        (run / "request.md").write_bytes((case / "preparation.md").read_bytes())
        source = run / "cycles" / "000"
        cycle = run / "cycles" / "001"
        candidate = {
            "candidate_id": "measured-method", "mechanism": "按重要性保留历史 KV 的方法",
            "causal_chain": ["识别关键历史 token", "在相同预算下保留关键表示"],
            "storage_design": "总预算不超过 stock 对照", "minimal_code_change": "在独立工作区实现保留策略",
        }
        atomic_write_jsonl(source / "candidates.jsonl", [candidate])
        selection_path = source / "selection.json"
        atomic_write_json(selection_path, control_envelope("selection", case.name, "test",
                          [path_ref(source / "candidates.jsonl")], selected_candidate_id=candidate["candidate_id"]))
        # Newer, unmeasured work must never become the final method.
        atomic_write_jsonl(run / "cycles" / "002" / "candidates.jsonl", [{
            "candidate_id": "unmeasured-method", "mechanism": "不得输出的未评测方案",
        }])
        plan_path = cycle / "experiment_plan.json"
        atomic_write_json(plan_path, control_envelope("experiment_plan", case.name, "test", [path_ref(selection_path)],
                          candidate_id=candidate["candidate_id"], evaluation_tier="formal"))
        metrics_path = cycle / "metrics.json"
        atomic_write_json(metrics_path, control_envelope("metrics", case.name, "test", [path_ref(plan_path)],
                          comparison_mode="full_30plus70_vecinfer_1bit", longbench={"score": 47.5},
                          gsm8k={"score": 60.0}, math500={"score": 35.0}, theoretical_storage_pass=True,
                          actual_storage_pass=True, baseline_reference={"longbench": {"score": 45.0},
                          "gsm8k": {"score": 5.0}, "math500": {"score": 3.0}}))
        registry_path = cycle / "verified_registry.json"
        atomic_write_json(registry_path, control_envelope("verified_registry", case.name, "test", [path_ref(metrics_path)]))
        decision_path = cycle / "decision.json"
        atomic_write_json(decision_path, control_envelope("decision", case.name, "test",
                          [path_ref(metrics_path), path_ref(registry_path)],
                          action="STOP" if achieved else "REFINE", rationale="本轮已完成有效测量"))
        terminal_path = run / "terminal.json"
        atomic_write_json(terminal_path, control_envelope("terminal", case.name, "test",
                          [path_ref(decision_path), path_ref(registry_path)],
                          final_candidate_id=candidate["candidate_id"],
                          termination_type="objective_met" if achieved else "cycle_budget_exhausted",
                          unresolved=[] if achieved else ["质量目标尚未达成"], reproduction_entrypoint="复现本次已记录实验"))
        return case, run, metrics_path

    def test_final_report_uses_terminal_candidate_including_promoted_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, run, _ = self._completed_run(Path(temporary))
            output = export_method(case, run)
            text = output.read_text()
            self.assertEqual(output, case / "final_method.md")
            self.assertIn("主要思想", text)
            self.assertIn("识别关键历史 token", text)
            self.assertIn("47.5", text)
            self.assertIn("已达到第二阶段的目标判定", text)
            self.assertNotIn("不得输出的未评测方案", text)

    def test_no_final_report_before_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = create_try(Path(temporary), VALUES)
            run = case / "runs" / case.name
            run.mkdir(parents=True)
            (run / "request.md").write_bytes((case / "preparation.md").read_bytes())
            with self.assertRaisesRegex(ValueError, "尚未结束"):
                export_method(case, run)
            self.assertFalse((case / "final_method.md").exists())

    def test_budget_stop_does_not_claim_objective_met(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, run, _ = self._completed_run(Path(temporary), achieved=False)
            text = export_method(case, run).read_text()
            self.assertIn("未记录目标达成", text)
            self.assertIn("质量目标尚未达成", text)

    def test_changed_evidence_cannot_overwrite_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, run, metrics_path = self._completed_run(Path(temporary))
            output = export_method(case, run)
            original = output.read_bytes()
            metrics = read_json(metrics_path)
            metrics["longbench"]["score"] = 99.0
            atomic_write_json(metrics_path, metrics)
            with self.assertRaisesRegex(ValueError, "哈希已变化"):
                export_method(case, run)
            self.assertEqual(output.read_bytes(), original)

    def test_changed_method_description_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, run, _ = self._completed_run(Path(temporary))
            candidates = run / "cycles" / "000" / "candidates.jsonl"
            atomic_write_jsonl(candidates, [{"candidate_id": "measured-method", "mechanism": "被替换的说明"}])
            with self.assertRaisesRegex(ValueError, "哈希已变化"):
                export_method(case, run)
            self.assertFalse((case / "final_method.md").exists())

    def test_research_completion_automatically_runs_output_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            case, run, _ = self._completed_run(root)
            with patch("kvresearch.navigator.Navigator") as navigator:
                navigator.return_value.next.return_value = Action("DONE", "terminal", "terminal", (), None)
                result = workflow.run_research(case, config.path)
            self.assertEqual(result["state"], "DONE")
            self.assertEqual(result["final_method"], str(case / "final_method.md"))
            self.assertTrue((case / "final_method.md").is_file())


if __name__ == "__main__":
    unittest.main()
