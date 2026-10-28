"""Execute a preparation-defined evaluator and score frozen paired records."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from kvresearch.config import Config
from kvresearch.remote import RemotePod
from kvresearch.research_data import checked_control
from kvresearch.util import atomic_write_json, atomic_write_jsonl, atomic_write_text, ensure_within, path_ref, read_json, read_jsonl, utcnow


class ResearchTarget(RemotePod):
    """Reuse token-scoped process cleanup, with no domain-specific runtime probe."""

    def __init__(self, config: Config, context: dict[str, Any]):
        self.hardware = context["hardware"]
        self.local = self.hardware.get("mode", "local") == "local"
        remote = {"python": self.hardware.get("python", sys.executable), **self.hardware}
        super().__init__(Config(config.path, {**config.data, "remote": remote}))

    def ssh_argv(self, command: str) -> list[str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        socket = self.hardware.get("control_socket")
        if socket and Path(socket).exists():
            argv.extend(["-S", str(socket)])
        argv.extend(["-p", str(self.hardware.get("port", 22)),
                     f"{self.hardware['user']}@{self.hardware['host']}", command])
        return argv

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        if self.local:
            return subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=timeout, check=False)
        return super().run(command, timeout=timeout)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def preflight(config: Config, run_dir: Path, output: Path) -> dict[str, Any]:
    context_path = run_dir / "research_context.json"
    context = checked_control(context_path, "research_context")
    target = ResearchTarget(config, context)
    hardware = context["hardware"]
    errors = []
    gpus = []
    try:
        probe = target.run(shlex.join([hardware.get("python", sys.executable), "-c", "import sys; print(sys.version)"]))
        if probe.returncode:
            errors.append(probe.stderr.strip() or "Python运行时不可用")
        if hardware.get("gpu_ids"):
            result = target.run("nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits")
            if result.returncode:
                errors.append(result.stderr.strip() or "无法读取GPU")
            else:
                for line in result.stdout.splitlines():
                    values = [part.strip() for part in line.split(",")]
                    if len(values) == 4:
                        gpus.append({"index": values[0], "uuid": values[1], "free_mib": int(values[2]), "utilization": int(values[3])})
                for gpu_id in hardware["gpu_ids"]:
                    device = next((gpu for gpu in gpus if str(gpu_id) in {gpu["index"], gpu["uuid"]}), None)
                    if device is None:
                        errors.append(f"用户指定GPU不可见：{gpu_id}")
                    elif device["free_mib"] < int(hardware.get("minimum_free_memory_mib", 0)) or device["utilization"] > int(hardware.get("maximum_utilization_percent", 100)):
                        errors.append(f"GPU未满足本次声明的资源条件：{gpu_id}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(str(exc))
    artifact = {"captured_at": utcnow(), "status": "complete", "available": not errors,
                "hardware": hardware, "gpus": gpus, "errors": errors, "inputs": [path_ref(context_path)]}
    atomic_write_json(output, artifact)
    return artifact


def validate_execution(plan: dict[str, Any], workspace: Path) -> None:
    ensure_within(Path(plan["workspace"]), workspace.parent)
    execution = plan.get("execution", {})
    command = execution.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise ValueError("评测计划 execution.command 必须是argv字符串数组")
    for token in ("{arm}", "{dataset}", "{output}", "{task_id}"):
        if not any(token in part for part in command):
            raise ValueError(f"评测入口必须接收锁定的 {token}")
    if not isinstance(execution.get("env", {}), dict):
        raise ValueError("execution.env 必须为对象")
    for key, value in execution.get("env", {}).items():
        if not isinstance(key, str) or not key.replace("_", "a").isalnum() or key[0].isdigit() or not isinstance(value, (str, int, float)):
            raise ValueError("评测环境变量名称或类型无效")


def _run_one(config: Config, run_dir: Path, plan: dict[str, Any], dataset: Path, task: str,
             arm: str, phase: str, destination: Path, gpu: Any = None) -> None:
    context = checked_control(run_dir / "research_context.json", "research_context")
    workspace = Path(plan["workspace"])
    validate_execution(plan, workspace)
    values = {"arm": arm, "dataset": str(dataset), "output": str(destination), "task_id": task,
              "workspace": str(workspace), "context": str(run_dir / "research_context.json"),
              "run": str(run_dir), "phase": phase, "python": context["hardware"].get("python", sys.executable)}
    command = []
    for part in plan["execution"]["command"]:
        for key, value in values.items():
            part = part.replace("{" + key + "}", value)
        command.append(part)
    env = {str(key): str(value) for key, value in plan["execution"].get("env", {}).items()}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_dir = destination.parent / "logs" / destination.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    env["KVRESEARCH_REMOTE_REGISTRY_DIR"] = str(destination.parent / "process_registry")
    env["KVRESEARCH_REMOTE_EXECUTION_UNIT"] = f"{phase}:{task}:{arm}"
    maximum = int(config.section("automation").get("command_timeout_seconds", 14400))
    timeout = min(maximum, int(plan["execution"].get("timeout_seconds", maximum)))
    if timeout <= 0:
        raise ValueError("实验超时必须为正数")
    atomic_write_json(log_dir / "command.json", {"argv": command, "cwd": str(workspace),
                      "environment_keys": sorted(env), "dataset": path_ref(dataset), "started_at": utcnow()})
    target = ResearchTarget(config, context)
    result = target.execute_argv(command, workspace, env, timeout=timeout)
    atomic_write_text(log_dir / "stdout.log", result.stdout)
    atomic_write_text(log_dir / "stderr.log", result.stderr)
    if result.returncode:
        raise RuntimeError(f"评测执行失败 {phase}/{task}/{arm}，退出码 {result.returncode}；日志 {log_dir}")
    if not destination.is_file():
        raise ValueError(f"评测没有发布逐题记录：{destination}")


def read_predictions(context: dict[str, Any], source: Path, output: Path, task: str) -> dict[str, dict[str, Any]]:
    expected = {row["sample_id"] for row in read_jsonl(source)}
    rows = read_jsonl(output)
    actual = [row.get("sample_id") for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"预测样本身份、数量或唯一性不符合固定分区：{output}")
    required_metrics = [metric["id"] for metric in context["metrics"] if task in metric["tasks"]]
    for row in rows:
        if row.get("status") != "complete":
            raise ValueError(f"逐题记录未完成：{output}/{row.get('sample_id')}")
        if any(not _finite((row.get("metrics") or {}).get(metric)) for metric in required_metrics):
            raise ValueError(f"逐题指标缺失或非有限：{output}")
        if any(not _finite((row.get("constraints") or {}).get(item["id"])) for item in context["constraints"]):
            raise ValueError(f"本次声明的预算测量不完整：{output}")
    return {row["sample_id"]: row for row in rows}


def assess(context: dict[str, Any], phases: list[Path], datasets: dict[str, dict[str, Path]]) -> dict[str, Any]:
    arm_rows: dict[str, dict[str, list[dict[str, Any]]]] = {"baseline": {}, "candidate": {}}
    refs = []
    violations = []
    for directory in phases:
        for task, source in datasets[directory.name].items():
            pair = {}
            for arm in arm_rows:
                path = directory / arm / f"{task}.jsonl"
                rows = read_predictions(context, source, path, task)
                arm_rows[arm].setdefault(task, []).extend(rows.values())
                pair[arm] = rows
                refs.append(path_ref(path))
            for sample_id, candidate in pair["candidate"].items():
                baseline = pair["baseline"][sample_id]
                for constraint in context["constraints"]:
                    value = candidate["constraints"][constraint["id"]]
                    limit = constraint.get("limit")
                    if limit is None:
                        limit = baseline["constraints"][constraint["id"]] * constraint["baseline_ratio"]
                    passed = {"<=": value <= limit, ">=": value >= limit, "==": value == limit}[constraint["operator"]]
                    if not passed:
                        violations.append({"constraint": constraint["id"], "sample_id": sample_id,
                                           "candidate": value, "limit": limit})
    quality = {}
    for metric in context["metrics"]:
        scores = {}
        for arm, tasks in arm_rows.items():
            groups = [[row["metrics"][metric["id"]] for row in tasks[task]] for task in metric["tasks"]]
            if any(not values for values in groups):
                raise ValueError(f"指标没有完整覆盖评测任务：{metric['id']}")
            if metric.get("aggregation", "macro") == "micro":
                values = [value for group in groups for value in group]
                scores[arm] = sum(values) / len(values)
            else:
                scores[arm] = sum(sum(group) / len(group) for group in groups) / len(groups)
        delta = scores["candidate"] - scores["baseline"]
        gain = delta if metric["direction"] == "maximize" else -delta
        quality[metric["id"]] = {"label": metric.get("label", metric["id"]), "direction": metric["direction"],
                                  **scores, "gain": gain, "strict_gain": gain > 0, "no_regression": gain >= 0}
    no_regression = all(metric["no_regression"] for metric in quality.values())
    any_gain = any(metric["strict_gain"] for metric in quality.values())
    return {"quality": quality, "constraints_pass": not violations, "constraint_violations": violations,
            "no_regressions": no_regression, "any_strict_gain": any_gain,
            "all_strict_gains": all(metric["strict_gain"] for metric in quality.values()), "inputs": refs}


def phase_sources(lock: dict[str, Any], partition: str) -> dict[str, Path]:
    return {task: Path(entry["partitions"][partition]["path"]) for task, entry in lock["tasks"].items()}


def run_phase(config: Config, run_dir: Path, plan: dict[str, Any], directory: Path,
              sources: dict[str, Path]) -> None:
    context = checked_control(run_dir / "research_context.json", "research_context")
    gpu_ids = context["hardware"].get("gpu_ids", [])
    workers = gpu_ids or [None] * max(1, int(context["hardware"].get("parallel_workers", 1)))
    queue: Queue = Queue()
    for task, dataset in sources.items():
        queue.put((task, dataset))

    def worker(gpu: Any) -> None:
        while True:
            try:
                task, dataset = queue.get_nowait()
            except Empty:
                return
            try:
                for arm in ("baseline", "candidate"):
                    destination = directory / arm / f"{task}.jsonl"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _run_one(config, run_dir, plan, dataset, task, arm, directory.name, destination, gpu)
                    read_predictions(context, dataset, destination, task)
            finally:
                queue.task_done()
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in workers]
        for future in futures:
            future.result()


def execute(config: Config, run_dir: Path, cycle_dir: Path, attempt_dir: Path) -> dict[str, Any]:
    context = checked_control(run_dir / "research_context.json", "research_context")
    lock = checked_control(run_dir / "datasets.lock.json", "datasets_lock")
    plan_path = cycle_dir / "experiment_plan.json"
    plan = checked_control(plan_path, "experiment_plan")
    implementation = cycle_dir / "implementation_manifest.json"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = attempt_dir / "run_manifest.json"
    atomic_write_json(manifest_path, {"status": "partial", "started_at": utcnow(),
                      "inputs": [path_ref(plan_path), path_ref(implementation), path_ref(run_dir / "datasets.lock.json")]})
    phases = ["development"] if plan["evaluation_tier"] == "development" else ["formal"]
    error = None
    gate = None
    try:
        primary = phases[0]
        run_phase(config, run_dir, plan, attempt_dir / primary, phase_sources(lock, primary))
        if primary == "formal":
            gate = assess(context, [attempt_dir / "formal"], {"formal": phase_sources(lock, "formal")})
            quality_pass = gate["all_strict_gains"] if context.get("success_rule") == "all_strict" else gate["no_regressions"] and gate["any_strict_gain"]
            gate["qualified_for_completion"] = quality_pass and gate["constraints_pass"]
            atomic_write_json(attempt_dir / "formal_gate.json", gate)
            if gate["qualified_for_completion"]:
                # Replay the fixed development partition under this exact final
                # implementation; never mix scores from an older implementation.
                for partition in ("remaining", "development"):
                    run_phase(config, run_dir, plan, attempt_dir / partition, phase_sources(lock, partition))
                    phases.append(partition)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {"status": "complete" if error is None else "failed", "exit_code": 0 if error is None else 1,
              "completed_at": utcnow(), "phases": phases, "error": error,
              "qualified_for_completion": bool(gate and gate.get("qualified_for_completion")),
              "inputs": [path_ref(manifest_path)]}
    atomic_write_json(attempt_dir / "exit.json", result)
    return result
