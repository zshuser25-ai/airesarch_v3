from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvresearch.config import Config
from kvresearch.remote import RemotePod
from kvresearch.stock_control import stock_control_env


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: int
    raw_output: Path | None = None
    gpu_index: int = 0
    gpu_uuid: str = ""
    task_name: str | None = None
    unit_id: str | None = None
    sample_start: int | None = None
    sample_end: int | None = None
    final_output: Path | None = None


class VecInferAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.remote = RemotePod(config)

    def prepare_datasets(self, run_dir: Path, tier: str) -> Path:
        target = run_dir / "runtime_datasets" / tier
        marker = target / ".complete.json"
        if marker.is_file():
            return target
        command = shlex.join(
            [
                self.config.section("remote")["python"],
                "-m",
                "kvresearch.data",
                "--config",
                str(self.config.path),
                "--run",
                str(run_dir),
                "--tier",
                tier,
            ]
        )
        result = self.remote.run(
            f"cd {shlex.quote(str(self.config.project_root))} && PYTHONDONTWRITEBYTECODE=1 {command}", timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(f"dataset preparation failed: {result.stderr}")
        return target

    def _base_env(self, gpu_index: int, dataset_root: Path, plan: dict[str, Any]) -> dict[str, str]:
        remote = self.config.section("remote")
        workspace = Path(plan["workspace"])
        plan_execution_env = plan.get("execution", {}).get("env", {})
        evaluation = self.config.section("evaluation")
        scored_math_cap = str(int(evaluation.get("math_quality_max_total_length", 2048)))
        smoke = str(plan_execution_env.get("KVRESEARCH_IMPLEMENTATION_SMOKE", "0")) == "1"
        expected_math_cap = "512" if smoke else scored_math_cap
        actual_math_cap = str(plan_execution_env.get("VECINFER_GENERATION_MAX_LENGTH", ""))
        if actual_math_cap != expected_math_cap:
            raise ValueError(
                "plan math quality cap drifted: "
                f"expected VECINFER_GENERATION_MAX_LENGTH={expected_math_cap}, got {actual_math_cap!r}"
            )
        expected_storage_length = str(int(evaluation["max_total_length"]))
        actual_storage_length = str(plan_execution_env.get("VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH", ""))
        if actual_storage_length != expected_storage_length:
            raise ValueError(
                "plan matched-storage length drifted: "
                f"expected VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH={expected_storage_length}, "
                f"got {actual_storage_length!r}"
            )
        extension_path = plan.get("execution", {}).get("extension_path")
        python_paths = [str(extension_path)] if extension_path else []
        python_paths.extend(str(path) for path in remote.get("runtime_pythonpath", []))
        python_paths.extend([str(workspace / "scripts" / "modeldb" / "bindings"), str(workspace)])
        torch_lib = "/opt/conda/lib/python3.11/site-packages/torch/lib"
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": ":".join(python_paths),
            "LD_LIBRARY_PATH": torch_lib,
            "KVRESEARCH_DATASETS_ROOT": str(dataset_root),
            "VECINFER_PRECISION_LABEL": str(plan.get("precision_label", "candidate-1bit")),
            "VECINFER_EXECUTION_BACKEND": "KVResearch isolated VecInfer candidate",
        }
        protected_control_env = stock_control_env(self.config)
        for key, value in plan_execution_env.items():
            if not key.startswith(("VECINFER_", "KVRESEARCH_")):
                raise ValueError(f"unsupported experiment environment variable: {key}")
            if key in protected_control_env and str(value) != protected_control_env[key]:
                raise ValueError(f"plan attempted to override host stock-control invariant: {key}")
            env[str(key)] = str(value)
        env.update(protected_control_env)
        return env

    def commands(self, run_dir: Path, cycle: str, attempt: str, gpus: list[dict[str, Any]], plan: dict[str, Any],
                 tier_override: str | None = None, raw_subdir: str = "raw",
                 dataset_root_override: Path | None = None) -> list[CommandSpec]:
        tier = tier_override or str(plan.get("evaluation_tier", "development"))
        dataset_root = Path(dataset_root_override).resolve() if dataset_root_override else self.prepare_datasets(run_dir, tier)
        workspace = Path(plan["workspace"]).resolve()
        attempt_dir = run_dir / "cycles" / cycle / "attempts" / attempt
        raw_dir = attempt_dir / raw_subdir
        raw_dir.mkdir(parents=True, exist_ok=True)
        python = self.config.section("remote")["vecinfer_python"]
        storage = self.config.section("storage")["vecinfer"]
        if tier == "development":
            tasks = list(self.config.section("evaluation")["development_tasks"])
        else:
            tasks = list(self.config.section("evaluation")["longbench_tasks"]) + list(self.config.section("evaluation")["math_tasks"])
        selected = plan.get("datasets")
        if selected:
            tasks = [task for task in tasks if task in selected]
        timeout = int(plan.get("timeout_seconds", self.config.section("automation")["command_timeout_seconds"]))
        commands: list[CommandSpec] = []
        if not gpus:
            raise ValueError("no approved GPU was supplied")
        formal_batches = list(self.config.section("evaluation").get("formal_batch_sizes", [1]))
        implementation_smoke = str(
            (plan.get("execution", {}).get("env", {}) or {}).get(
                "KVRESEARCH_IMPLEMENTATION_SMOKE", "0"
            )
        ) == "1"
        # The implementation gate intentionally supplies one locked row per
        # task.  It must therefore remain batch 1 even for a formal plan;
        # formal scored execution still starts at the configured highest
        # batch and uses the normal OOM backoff schedule.
        initial_batch = (
            int(plan.get("batch_size", 1))
            if implementation_smoke or tier not in {"formal", "formal_remaining"}
            else int(formal_batches[0])
        )
        for position, task in enumerate(tasks):
            gpu = gpus[position % len(gpus)]
            env = self._base_env(int(gpu["index"]), dataset_root, plan)
            dataset_arg = "math" if task == "math500" else task
            sink = storage["math_sink_tokens"] if task in {"gsm8k", "math500"} else storage["longbench_sink_tokens"]
            output = raw_dir / f"{task}.jsonl"
            argv = [
                python, "-m", "scripts.modeldb.main_pq",
                "-f", "llama-kvresearch.json",
                "--dataset", dataset_arg,
                "--cluster_dataset", "qasper",
                "-M_key", str(storage["key_subspaces"]),
                "-M_value", str(storage["value_subspaces"]),
                "--nbits_key", str(storage["key_bits_per_code"]),
                "--nbits_value", str(storage["value_bits_per_code"]),
                "--sink_tokens", str(sink),
                "--batch_size", str(initial_batch),
                "-m", "--half", "--seed", str(self.config.section("evaluation")["seed"]),
                "--save_path", str(output), "-p", "evaluation",
            ]
            task_env = dict(env)
            if task in {"gsm8k", "math500"}:
                task_env["VECINFER_MATH_DO_SAMPLE"] = "1"
            commands.append(CommandSpec(f"generate-{task}", argv, workspace, task_env, timeout, output,
                                        int(gpu["index"]), str(gpu.get("uuid", ""))))
        return commands
