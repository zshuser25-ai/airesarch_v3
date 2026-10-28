"""Artifact-driven research whose scientific configuration comes from Markdown."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from kvresearch.config import Config
from kvresearch.navigator import Action, _cycle_budget_reached
from kvresearch.research_agent import implementation_files, run_agent, snapshot_fields, validate_result
from kvresearch.research_data import checked_control, freeze_datasets, validate_context
from kvresearch.research_execution import ResearchTarget, assess, execute, phase_sources, preflight, run_phase, validate_execution
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl, atomic_write_text, ensure_within, latest_numbered_dir, path_ref, read_json, read_jsonl, sha256_file


def enabled(config: Config) -> bool:
    return config.data.get("research", {}).get("engine") == "preparation"


def initialize(config: Config, request: str, run_id: str, preparation: Path | None = None) -> Path:
    if not request.strip() or not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("研究请求与run_id无效")
    run = config.runs_dir / run_id
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    atomic_write_text(run / "request.md", request.rstrip() + "\n")
    refs = [path_ref(run / "request.md"), path_ref(config.path), path_ref(config.catalog_path)]
    if preparation:
        refs.append(path_ref(preparation))
    atomic_write_json(run / "run_manifest.json", control_envelope("run_manifest", run_id, "kvresearch.research", refs,
                      engine="preparation", project_root=str(config.project_root), preparation=str(preparation) if preparation else None))
    atomic_write_json(run / "inputs.lock.json", control_envelope("inputs_lock", run_id, "kvresearch.research", refs,
                      stage="preparation_defined", scientific_context=str(run / "research_context.json"),
                      dataset_lock=str(run / "datasets.lock.json")))
    return run


def _valid(path: Path, kind: str, *, bindings: list[Path] | None = None) -> bool:
    try:
        value = checked_control(path, kind)
        for binding in bindings or []:
            refs = {ref["path"]: ref["sha256"] for ref in value.get("inputs", [])}
            if refs.get(str(binding.resolve())) != sha256_file(binding):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _agent(run: Path, step: str, inputs: list[Path], output: Path) -> Action:
    manifest = read_json(run / "run_manifest.json")
    original = Path(manifest["preparation"]) if manifest.get("preparation") else run / "request.md"
    base = [original, run / "request.md", run / "inputs.lock.json"]
    for path in (run / "research_context.json", run / "datasets.lock.json"):
        if path.is_file() and path not in inputs and path != output:
            base.append(path)
    paths = list(dict.fromkeys(base + inputs))
    return Action("READY", step, "agent", tuple(str(path) for path in paths), str(output))


def _command(config: Config, run: Path, step: str, output: Path, cycle: Path | None = None, attempt: Path | None = None) -> Action:
    argv = ["kvresearch", "--config", str(config.path), step, "--run", str(run)]
    if cycle:
        argv += ["--cycle", cycle.name]
    if attempt:
        argv += ["--attempt", attempt.name]
    return Action("READY", step, "command", (), str(output), tuple(argv))


def _selection_for_plan(plan: dict[str, Any], run: Path) -> Path:
    selection = next((Path(ref["path"]) for ref in plan.get("inputs", []) if Path(ref["path"]).name == "selection.json"), None)
    if selection is None:
        raise ValueError("实验计划未绑定候选选择")
    return ensure_within(selection, run)


def _implementation_current(path: Path) -> bool:
    if not _valid(path, "implementation_manifest"):
        return False
    value = read_json(path)
    workspace = Path(value["workspace"])
    try:
        return implementation_files(workspace) == value.get("file_hashes")
    except (OSError, ValueError):
        return False


def next_action(config: Config, run: Path) -> Action:
    for name, kind in (("run_manifest.json", "run_manifest"), ("inputs.lock.json", "inputs_lock")):
        try:
            checked_control(run / name, kind)
        except (OSError, ValueError, KeyError) as exc:
            return Action("INVALID", "locked_inputs", "repair_input", (), None, notes=str(exc))
    terminal = run / "terminal.json"
    if terminal.is_file():
        return Action("DONE" if _valid(terminal, "terminal") else "INVALID", "terminal", "terminal", (str(terminal),), None)
    context = run / "research_context.json"
    if not _valid(context, "research_context"):
        return _agent(run, "research_intake", [], context)
    validate_context(read_json(context))
    lock = run / "datasets.lock.json"
    if not lock.is_file():
        return _command(config, run, "freeze-data", lock)
    try:
        checked_control(lock, "datasets_lock")
    except (OSError, ValueError, KeyError) as exc:
        return Action("INVALID", "frozen_datasets", "repair_input", (), None, notes=str(exc))
    literature = run / "evidence" / "literature.jsonl"
    try:
        validate_result("literature", read_jsonl(literature), [])
    except (OSError, ValueError, KeyError):
        return _agent(run, "literature", [], literature)
    synthesis = run / "evidence" / "synthesis.json"
    if not _valid(synthesis, "evidence_synthesis"):
        return _agent(run, "evidence_synthesis", [literature], synthesis)
    cycle = latest_numbered_dir(run / "cycles") or run / "cycles" / "000"
    # Completed cycles advance monotonically. A PROMOTE carries the measured
    # method into formal evaluation without another candidate tournament.
    if _valid(cycle / "decision.json", "decision") and (cycle / "lessons.jsonl").is_file():
        decision = read_json(cycle / "decision.json")
        if decision["action"] == "STOP" or _cycle_budget_reached(int(config.section("automation").get("max_cycles", 12)), int(cycle.name)):
            return _command(config, run, "finalize", terminal, cycle)
        previous = cycle
        cycle = cycle.parent / f"{int(cycle.name) + 1:03d}"
    else:
        previous = cycle.parent / f"{int(cycle.name) - 1:03d}" if int(cycle.name) else None
    promoted = bool(previous and _valid(previous / "decision.json", "decision")
                    and read_json(previous / "decision.json").get("action") == "PROMOTE")
    if promoted:
        old_plan = checked_control(previous / "experiment_plan.json", "experiment_plan")
        selection = _selection_for_plan(old_plan, run)
    else:
        diagnosis = cycle / "problem_diagnosis.json"
        prior = [previous / name for name in ("decision.json", "lessons.jsonl") if previous and (previous / name).is_file()]
        if not _valid(diagnosis, "problem_diagnosis"):
            return _agent(run, "problem_diagnosis", [synthesis] + prior, diagnosis)
        candidates = cycle / "candidates.jsonl"
        try:
            validate_result("candidates", read_jsonl(candidates), [])
        except (OSError, ValueError, KeyError):
            return _agent(run, "candidates", [diagnosis, synthesis] + prior, candidates)
        selection = cycle / "selection.json"
        if not _valid(selection, "selection"):
            return _agent(run, "selection", [diagnosis, candidates] + prior, selection)
    plan = cycle / "experiment_plan.json"
    if not _valid(plan, "experiment_plan"):
        inputs = [selection]
        if promoted:
            inputs += [previous / name for name in ("decision.json", "experiment_plan.json", "implementation_manifest.json")]
        return _agent(run, "experiment_plan", inputs, plan)
    workspace = cycle / "workspace"
    if not workspace.is_dir():
        return _command(config, run, "prepare-workspace", workspace, cycle)
    implementation = cycle / "implementation_manifest.json"
    if not _implementation_current(implementation):
        inputs = [plan, selection]
        if promoted:
            inputs.append(previous / "implementation_manifest.json")
        return _agent(run, "implementation_manifest", inputs, implementation)
    validation = cycle / "validation.json"
    if not _valid(validation, "validation", bindings=[plan, implementation]):
        return _command(config, run, "validate-cycle", validation, cycle)
    validated = read_json(validation)
    if not validated.get("all_hard_checks_pass"):
        if validated.get("resource_wait"):
            if os.environ.pop("KVRESEARCH_RESUME_REMOTE_ONCE", "") in {"1", "true"}:
                return _command(config, run, "validate-cycle", validation, cycle)
            return Action("WAIT", "remote_resources", "terminal", (str(validation),), None, notes="检查用户声明的硬件后显式恢复")
        return _agent(run, "implementation_repair", [plan, selection, implementation, validation], implementation)
    attempt = latest_numbered_dir(cycle / "attempts") or cycle / "attempts" / "000"
    manifest = attempt / "run_manifest.json"
    if manifest.is_file():
        executed = read_json(manifest)
        refs = {ref["path"]: ref["sha256"] for ref in executed.get("inputs", [])}
        if refs.get(str(implementation.resolve())) != sha256_file(implementation) or refs.get(str(plan.resolve())) != sha256_file(plan):
            attempt = attempt.parent / f"{int(attempt.name) + 1:03d}"
    preflight_path = attempt / "preflight.json"
    if not preflight_path.is_file():
        return _command(config, run, "preflight", preflight_path, cycle, attempt)
    if not read_json(preflight_path).get("available"):
        if os.environ.pop("KVRESEARCH_RESUME_REMOTE_ONCE", "") in {"1", "true"}:
            attempt = attempt.parent / f"{int(attempt.name) + 1:03d}"
            return _command(config, run, "preflight", attempt / "preflight.json", cycle, attempt)
        return Action("WAIT", "remote_resources", "terminal", (str(preflight_path),), None)
    exit_path = attempt / "exit.json"
    if not exit_path.is_file():
        return _command(config, run, "monitor" if (attempt / "run_manifest.json").is_file() else "execute", exit_path, cycle, attempt)
    if read_json(exit_path).get("status") != "complete":
        return _agent(run, "execution_repair", [plan, selection, implementation, exit_path], implementation)
    metrics = cycle / "metrics.json"
    if not _valid(metrics, "metrics", bindings=[implementation]):
        return _command(config, run, "aggregate", metrics, cycle, attempt)
    registry = cycle / "verified_registry.json"
    if not _valid(registry, "verified_registry"):
        return _command(config, run, "verify", registry, cycle)
    diagnosis = cycle / "experiment_diagnosis.json"
    if not _valid(diagnosis, "experiment_diagnosis"):
        return _command(config, run, "diagnose", diagnosis, cycle)
    decision = cycle / "decision.json"
    if not _valid(decision, "decision"):
        return _command(config, run, "decide", decision, cycle)
    return _command(config, run, "learn", cycle / "lessons.jsonl", cycle)


def _artifact(run: Path, path: Path, kind: str, inputs: list[Path], **domain: Any) -> dict[str, Any]:
    value = control_envelope(kind, run.name, "kvresearch.research", [path_ref(item) for item in inputs], **domain)
    atomic_write_json(path, value)
    return value


def validate_cycle(config: Config, run: Path, cycle: Path) -> dict[str, Any]:
    plan_path, implementation_path = cycle / "experiment_plan.json", cycle / "implementation_manifest.json"
    plan = checked_control(plan_path, "experiment_plan")
    implementation = checked_control(implementation_path, "implementation_manifest")
    root = cycle / "validation_runs" / implementation["implementation_sha256"]
    context = checked_control(run / "research_context.json", "research_context")
    lock = checked_control(run / "datasets.lock.json", "datasets_lock")
    error = None
    resource_wait = False
    try:
        workspace = ensure_within(Path(plan["workspace"]), cycle)
        if workspace != cycle / "workspace" or Path(implementation["workspace"]) != workspace:
            raise ValueError("实现与计划的工作区不一致")
        if set(plan["datasets"]) != set(lock["tasks"]):
            raise ValueError("计划必须覆盖第一阶段指定并已固定的所有数据集")
        validate_execution(plan, workspace)
        if not _implementation_current(implementation_path):
            raise ValueError("实现快照哈希已变化")
        for name in implementation["file_hashes"]:
            if name.endswith(".py"):
                compile((workspace / name).read_text(encoding="utf-8"), name, "exec")
        resources = preflight(config, run, root / "preflight.json")
        if not resources["available"]:
            resource_wait = True
            raise ValueError("; ".join(resources["errors"]))
        target = ResearchTarget(config, context)
        for index, command in enumerate(implementation.get("build_commands", [])):
            if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
                raise ValueError("构建命令必须为argv数组")
            result = target.execute_argv(command, workspace, {}, int(config.section("automation").get("command_timeout_seconds", 14400)))
            atomic_write_text(root / f"build-{index}.log", result.stdout + result.stderr)
            if result.returncode:
                raise ValueError(f"构建失败：{root / f'build-{index}.log'}")
        built = snapshot_fields(workspace)
        if built["file_hashes"] != implementation["file_hashes"]:
            implementation.update(built)
            atomic_write_json(implementation_path, implementation)
        sources = {}
        for task, entry in lock["tasks"].items():
            first = read_jsonl(Path(entry["partitions"]["development"]["path"]))[0]
            path = root / "smoke_inputs" / f"{task}.jsonl"
            atomic_write_jsonl(path, [first])
            sources[task] = path
        run_phase(config, run, plan, root / "smoke", sources)
        # Liveness and complete measurements are required; smoke quality and
        # measured budget excess do not establish a research conclusion.
        assess(context, [root / "smoke"], {"smoke": sources})
        if not _implementation_current(implementation_path):
            raise ValueError("冒烟评测修改了实现快照；评测输出和缓存应写入运行目录")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _artifact(run, cycle / "validation.json", "validation", [plan_path, implementation_path, run / "datasets.lock.json"],
                     all_hard_checks_pass=error is None, resource_wait=resource_wait, error=error, evidence=str(root))


def aggregate(config: Config, run: Path, cycle: Path, attempt: Path) -> dict[str, Any]:
    context = checked_control(run / "research_context.json", "research_context")
    lock = checked_control(run / "datasets.lock.json", "datasets_lock")
    exit_data = read_json(attempt / "exit.json")
    phases = exit_data["phases"]
    result = assess(context, [attempt / phase for phase in phases], {phase: phase_sources(lock, phase) for phase in phases})
    full = set(phases) == {"development", "formal", "remaining"}
    source_inputs = [cycle / "experiment_plan.json", cycle / "implementation_manifest.json", run / "datasets.lock.json", attempt / "exit.json"]
    quality = result.pop("quality")
    refs = result.pop("inputs")
    value = _artifact(run, cycle / "metrics.json", "metrics", source_inputs,
                      quality=quality, **result, comparison_mode="full_5_30_65" if full else phases[0],
                      coverage={"partitions": phases, "full_dataset": full}, raw_sources=refs)
    value["inputs"] += refs
    atomic_write_json(cycle / "metrics.json", value)
    return value


def decide(config: Config, run: Path, cycle: Path) -> dict[str, Any]:
    context = checked_control(run / "research_context.json", "research_context")
    metrics = checked_control(cycle / "metrics.json", "metrics")
    plan = checked_control(cycle / "experiment_plan.json", "experiment_plan")
    selection = checked_control(_selection_for_plan(plan, run), "selection")
    candidates_path = next(Path(ref["path"]) for ref in selection["inputs"] if Path(ref["path"]).name == "candidates.jsonl")
    candidate = next(row for row in read_jsonl(candidates_path) if row["candidate_id"] == plan["candidate_id"])
    family = candidate["mechanism_family"]
    previous_decisions = []
    for index in range(int(cycle.name)):
        path = cycle.parent / f"{index:03d}" / "decision.json"
        if _valid(path, "decision"):
            previous_decisions.append(read_json(path))
    active = {item["mechanism_family"]: item for decision in previous_decisions for item in decision.get("cooldown", {}).get("active_families", [])
              if int(item["until_cycle"]) >= int(cycle.name)}
    same = next((decision for decision in reversed(previous_decisions) if decision.get("mechanism_family") == family and decision.get("evaluation_tier") == plan["evaluation_tier"]), None)
    small = False
    if same:
        deltas = {}
        for metric in context["metrics"]:
            now = metrics["quality"][metric["id"]]
            old = same.get("quality", {}).get(metric["id"])
            if old is None:
                continue
            delta = now["candidate"] - old["candidate"]
            deltas[metric["id"]] = delta if metric["direction"] == "maximize" else -delta
        meaningful = all(value >= 0 for value in deltas.values()) and any(
            deltas.get(metric["id"], float("-inf")) > 0 and deltas[metric["id"]] >= float(metric.get("meaningful_gain", 0)) for metric in context["metrics"])
        small = not meaningful
    streak = int(same.get("cooldown", {}).get("small_gain_streak", 0)) + 1 if same and small else 0
    triggered = streak >= 2
    if triggered:
        active[family] = {"mechanism_family": family, "until_cycle": int(cycle.name) + 2}
    full = metrics["coverage"]["full_dataset"]
    quality_ok = metrics["all_strict_gains"] if context.get("success_rule") == "all_strict" else metrics["no_regressions"] and metrics["any_strict_gain"]
    if not metrics["constraints_pass"]:
        action, rationale = "REFINE", "真实测量未满足本次用户定义的资源或预算约束"
    elif plan["evaluation_tier"] == "formal" and full and quality_ok:
        action, rationale = "STOP", "全量5%+30%+65%的逐题结果满足本次科研目标与约束"
    elif plan["evaluation_tier"] == "development" and metrics["no_regressions"] and metrics["any_strict_gain"]:
        action, rationale = "PROMOTE", "固定5%开发集取得严格收益且所有指标不下降，进入固定30%正式评测"
    elif plan["evaluation_tier"] == "formal" and not metrics["any_strict_gain"]:
        action, rationale = "ARCHIVE", "正式测量没有任何指标严格提升，归档本候选"
    else:
        action, rationale = "REFINE", "有效测量尚未满足本次科研目标，保留证据定向优化"
    consecutive = 1
    for decision in reversed(previous_decisions):
        if decision.get("action") in {"PIVOT", "PROMOTE", "ARCHIVE", "STOP"}:
            break
        consecutive += 1
    if action == "REFINE" and (triggered or consecutive >= 3):
        action, rationale = "PIVOT", "同族收益进入冷却或连续三个有效研究轮次未达到晋级条件，探索不同机制"
    return _artifact(run, cycle / "decision.json", "decision", [cycle / name for name in ("metrics.json", "verified_registry.json", "experiment_diagnosis.json")],
                     action=action, rationale=rationale, candidate_id=plan["candidate_id"], mechanism_family=family,
                     quality=metrics["quality"], evaluation_tier=plan["evaluation_tier"],
                     cooldown={"triggered": triggered, "small_gain_streak": streak, "active_families": list(active.values())})


def perform(config: Config, run: Path, action: Action) -> dict[str, Any]:
    output = Path(action.output) if action.output else None
    if action.kind == "agent":
        return run_agent(config, run, action.step, [Path(path) for path in action.inputs], output)
    step = action.step
    if step == "freeze-data":
        return freeze_datasets(config, run)
    # Cycle and attempt positions come from the host-created output path.
    relative = output.relative_to(run)
    cycle = run / "cycles" / relative.parts[1] if relative.parts[0] == "cycles" else None
    attempt = output.parent if cycle and "attempts" in relative.parts else None
    if step == "prepare-workspace":
        output.mkdir(parents=True, exist_ok=False)
        return {"status": "complete", "workspace": str(output)}
    if step == "validate-cycle":
        return validate_cycle(config, run, cycle)
    if step == "preflight":
        return preflight(config, run, output)
    if step == "execute":
        return execute(config, run, cycle, attempt)
    if step == "monitor":
        context = checked_control(run / "research_context.json", "research_context")
        target = ResearchTarget(config, context)
        registries = list(attempt.rglob("process_registry"))
        if not all(target.cleanup_registry(path) for path in registries):
            raise RuntimeError("中断执行的进程组尚未确认清理，不能启动重叠实验")
        result = {"status": "failed", "exit_code": 1, "error": "执行进程中断；已清理登记的进程组，需同周期修复和新attempt"}
        atomic_write_json(output, result)
        return result
    if step == "aggregate":
        return aggregate(config, run, cycle, latest_numbered_dir(cycle / "attempts"))
    if step == "verify":
        metrics = checked_control(cycle / "metrics.json", "metrics")
        return _artifact(run, output, "verified_registry", [cycle / "metrics.json"],
                         values=[{"id": key, **value, "source": path_ref(cycle / "metrics.json"), "json_pointer": f"/quality/{key}"}
                                 for key, value in metrics["quality"].items()])
    if step == "diagnose":
        metrics = checked_control(cycle / "metrics.json", "metrics")
        return _artifact(run, output, "experiment_diagnosis", [cycle / "metrics.json", cycle / "verified_registry.json"],
                         measurement_valid=True, deficiencies=metrics["constraint_violations"])
    if step == "decide":
        return decide(config, run, cycle)
    if step == "learn":
        decision = checked_control(cycle / "decision.json", "decision")
        row = {"lesson_id": f"{run.name}-{cycle.name}", "source_cycle": cycle.name,
               "decision": decision["action"], "rationale": decision["rationale"], "evidence": path_ref(cycle / "decision.json")}
        atomic_write_jsonl(output, [row])
        memory = run.parent.parent / "memory" / "lessons.jsonl"
        prior = read_jsonl(memory) if memory.is_file() else []
        atomic_write_jsonl(memory, [old for old in prior if old.get("lesson_id") != row["lesson_id"]] + [row])
        return {"status": "complete", "lessons": [row]}
    if step == "finalize":
        # The output path is terminal.json, so obtain the final completed cycle.
        cycle = latest_numbered_dir(run / "cycles")
        plan = checked_control(cycle / "experiment_plan.json", "experiment_plan")
        decision = checked_control(cycle / "decision.json", "decision")
        return _artifact(run, output, "terminal", [cycle / "decision.json", cycle / "verified_registry.json"],
                         termination_type="objective_met" if decision["action"] == "STOP" else "cycle_budget_exhausted",
                         final_candidate_id=plan["candidate_id"], unresolved=[] if decision["action"] == "STOP" else [decision["rationale"]],
                         reproduction_entrypoint=f"python3 workflow.py run --try {run.name}")
    raise ValueError(f"未知通用研究步骤：{step}")


def run_loop(config: Config, run: Path, max_steps: int, report_dir: Path) -> dict[str, Any]:
    from kvresearch.reporting import write_stage_report

    completed = 0
    for _ in range(max_steps):
        action = next_action(config, run)
        if action.kind in {"terminal", "repair_input"}:
            break
        try:
            outcome = perform(config, run, action)
        except Exception as exc:
            # A scorer/publication defect is technical, never a scientific loss.
            if action.step == "aggregate":
                attempt = latest_numbered_dir(Path(action.output).parent / "attempts")
                exit_data = read_json(attempt / "exit.json")
                exit_data.update(status="failed", error=f"aggregation: {type(exc).__name__}: {exc}")
                atomic_write_json(attempt / "exit.json", exit_data)
                continue
            raise
        if action.kind == "agent" and outcome.get("status") != "complete":
            return {"state": "NEEDS_INPUT" if outcome.get("status") == "needs_input" else "AGENT_INCOMPLETE",
                    "step": action.step, "run_dir": str(run), "outcome": outcome}
        write_stage_report(report_dir, run, action, outcome)
        print(f"第二阶段：{action.step} 已执行。", flush=True)
        completed += 1
    final = next_action(config, run)
    result = {"state": final.state, "step": final.step, "notes": final.notes,
              "run_dir": str(run), "completed_steps": completed}
    if final.state == "DONE":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from stage3_output import export_method
        result["final_method"] = str(export_method(run.parent.parent, run))
    return result
