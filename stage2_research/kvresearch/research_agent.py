"""Isolated AI stages for preparation-driven, domain-independent research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kvresearch.agent import _run_agent_command
from kvresearch.codex_backend import MAIN_AGENT_MODEL, MAIN_AGENT_REASONING_EFFORT, MAIN_AGENT_SERVICE_TIER
from kvresearch.config import Config
from kvresearch.research_data import checked_control, validate_context
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_text, ensure_within, path_ref, read_json, read_jsonl, sha256_file


RUNTIME = {"context_isolation": "fork_turns_none", "model": MAIN_AGENT_MODEL,
           "reasoning_effort": MAIN_AGENT_REASONING_EFFORT, "service_tier": MAIN_AGENT_SERVICE_TIER}
WORKERS = {
    "literature": ("closest_methods", "alternative_mechanisms", "evaluation_protocol", "systems_feasibility"),
    "evidence_synthesis": ("scientific_evidence", "evaluation_evidence", "engineering_evidence"),
    "problem_diagnosis": ("quality_bottleneck", "mechanism_bottleneck", "implementation_bottleneck"),
    "candidates": ("independent_proposal_1", "independent_proposal_2", "independent_proposal_3"),
    "selection": ("scientific_reviewer", "constraint_reviewer", "engineering_reviewer"),
}
REQUIRED = {
    "evidence_synthesis": ("known_methods", "bottlenecks", "reusable_mechanisms", "rejected_directions"),
    "problem_diagnosis": ("bottleneck", "observations", "evidence", "error_sources", "falsification", "exclusions"),
    "selection": ("selected_candidate_id", "ranking", "rejected", "evidence"),
    "experiment_plan": ("candidate_id", "evaluation_tier", "datasets", "workspace", "execution", "code_scope", "decision_rules"),
    "implementation_manifest": ("workspace", "implementation_notes"),
    "implementation_repair": ("workspace", "implementation_notes"),
    "execution_repair": ("workspace", "implementation_notes"),
}


def implementation_files(workspace: Path) -> dict[str, str]:
    files = [path for path in sorted(workspace.rglob("*")) if path.is_file() and
             not any(part in {".git", "__pycache__", "logs"} for part in path.relative_to(workspace).parts)]
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("实现快照需要实际文件，不能用软链接替代候选代码")
    return {str(path.relative_to(workspace)): sha256_file(path) for path in files}


def snapshot_fields(workspace: Path) -> dict[str, Any]:
    hashes = implementation_files(workspace)
    return {"file_hashes": hashes,
            "implementation_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def validate_result(step: str, value: Any, inputs: list[Path]) -> None:
    if step == "research_intake":
        validate_context(value)
        return
    if step == "literature":
        if not isinstance(value, list) or not value:
            raise ValueError("文献产物不能为空")
        for row in value:
            if not all(row.get(key) for key in ("paper_id", "title", "source_url", "relevance")):
                raise ValueError("文献需要可追溯的标识、来源和相关性")
        return
    if step == "candidates":
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("必须输出三个科研候选，不包含基线或对照候选")
        if {row.get("worker_id") for row in value} != {"worker-00", "worker-01", "worker-02"}:
            raise ValueError("三个候选必须各自来自一个独立worker")
        if len({row.get("candidate_id") for row in value}) != 3:
            raise ValueError("候选ID必须唯一")
        for row in value:
            if row.get("candidate_kind") != "research_proposal":
                raise ValueError("每个候选都必须是可选择的科研方案，不能是stock_control或baseline")
            for key in ("candidate_id", "mechanism_family", "mechanism", "causal_chain", "budget_design", "minimal_code_change", "falsification"):
                if not row.get(key):
                    raise ValueError(f"候选缺少 {key}")
        return
    for field in REQUIRED[step]:
        if field not in value:
            raise ValueError(f"{step} 缺少 {field}")
    if step == "selection":
        candidates = next(path for path in inputs if path.name == "candidates.jsonl")
        ids = {row["candidate_id"] for row in read_jsonl(candidates)}
        ranking = value["ranking"]
        if len(ranking) != 3 or set(ranking) != ids or ranking[0] != value["selected_candidate_id"]:
            raise ValueError("评审必须对三个真实候选排序，第一项为选中候选")
        cycle = int(candidates.parent.name)
        blocked = {item["mechanism_family"] for path in inputs if path.name == "decision.json"
                   for item in checked_control(path, "decision").get("cooldown", {}).get("active_families", [])
                   if int(item["until_cycle"]) >= cycle}
        selected = next(row for row in read_jsonl(candidates) if row["candidate_id"] == ranking[0])
        if selected["mechanism_family"] in blocked:
            raise ValueError("选中的机制族仍处于冷却期")
    if step == "experiment_plan":
        if value["evaluation_tier"] not in {"development", "formal"}:
            raise ValueError("评测阶段必须是development或formal")
        selection_path = next(path for path in inputs if path.name == "selection.json")
        selection = checked_control(selection_path, "selection")
        if value["candidate_id"] != selection["selected_candidate_id"]:
            raise ValueError("实验计划必须实现被选中的候选")
        for path in inputs:
            if path.name == "decision.json" and checked_control(path, "decision").get("action") == "PROMOTE":
                if value["evaluation_tier"] != "formal":
                    raise ValueError("开发晋级后必须直接进入正式评测并保留同一方法")
        if not isinstance(value["datasets"], list) or not value["datasets"]:
            raise ValueError("计划必须声明固定任务列表")


def run_agent(config: Config, run_dir: Path, step: str, inputs: list[Path], output: Path) -> dict[str, Any]:
    previous_revision = 0
    if output.is_file() and step in {"implementation_manifest", "implementation_repair", "execution_repair"}:
        previous_revision = int(read_json(output).get("revision", 0))
    common_path = Path(__file__).parent / "prompts" / "general_research.md"
    common = common_path.read_text(encoding="utf-8")
    refs = [path_ref(path) for path in inputs]
    fingerprint = hashlib.sha256(json.dumps({"step": step, "inputs": refs, "output": str(output),
                                "runtime": RUNTIME, "prompt": common, "workers": WORKERS.get(step)}, sort_keys=True).encode()).hexdigest()
    task_id = f"{step}-{fingerprint[:12]}"
    directory = run_dir / "agent_tasks" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    assignments = []
    for index, role in enumerate(WORKERS.get(step, ())):
        assignments.append({"worker_id": f"worker-{index:02d}", "role": role,
                            "output": str(directory / "workers" / f"worker-{index:02d}.json")})
    if assignments:
        (directory / "workers").mkdir(exist_ok=True)
    prompt = common + f"\n\nCurrent step: {step}\nOutput: {output}\n"
    prompt += "Inputs (read every file):\n" + "\n".join(f"- {path}" for path in inputs)
    prompt += "\n\nRuntime: " + json.dumps(RUNTIME)
    if assignments:
        prompt += (
            f"\nSpawn exactly {len(assignments)} workers concurrently, using fork_turns='none', "
            f"model='{MAIN_AGENT_MODEL}', reasoning_effort='{MAIN_AGENT_REASONING_EFFORT}'. "
            "Inherit the host's fast service tier; do not pass an unsupported service_tier spawn argument. "
            "Give every worker these input paths, its assigned objective, exact output path and this contract. "
            "Each worker writes only its own JSON: worker_id, role, status='complete', agent_runtime="
            + json.dumps(RUNTIME) + ", findings (array), evidence (array), recommendation. "
            "Wait for all workers and synthesize the sole formal output yourself. Reuse a prior worker only if its identity, "
            "runtime and status are valid for this exact task fingerprint.\nAssignments:\n"
            + json.dumps(assignments, ensure_ascii=False, indent=2)
        )
    prompt_path = directory / "prompt.md"
    atomic_write_text(prompt_path, prompt)
    atomic_write_json(directory / "task.json", {"task_id": task_id, "step": step, "inputs": refs,
                                               "output": str(output), "workers": assignments, "runtime": RUNTIME})
    template = config.section("automation").get("agent_command", [])
    if not template:
        return {"status": "needs_agent_backend", "task_packet": str(directory / "task.json")}
    substitutions = {"prompt": str(prompt_path), "run": str(run_dir), "task": str(directory / "task.json"),
                     "output": str(output), "max_parallel_subagents": str(config.section("automation").get("max_parallel_subagents", 4))}
    argv = [str(token).format(**substitutions) for token in template]
    result, attempts = _run_agent_command(argv, config.project_root,
        transient_retries=int(config.section("automation").get("agent_transient_retries", 6)),
        initial_backoff_seconds=float(config.section("automation").get("agent_retry_initial_seconds", 15)),
        max_backoff_seconds=60)
    for index, attempt in enumerate(attempts):
        atomic_write_text(directory / f"parent-{index:02d}.stdout.log", attempt.stdout)
        atomic_write_text(directory / f"parent-{index:02d}.stderr.log", attempt.stderr)
    if result.returncode:
        return {"status": "agent_failed", "task_packet": str(directory / "task.json"), "returncode": result.returncode}
    try:
        workers = []
        for assignment in assignments:
            path = Path(assignment["output"])
            worker = read_json(path)
            if any(worker.get(key) != assignment[key] for key in ("worker_id", "role")) or worker.get("status") != "complete" or worker.get("agent_runtime") != RUNTIME:
                raise ValueError(f"worker身份、状态或运行契约不合格：{path}")
            if not isinstance(worker.get("findings"), list) or not isinstance(worker.get("evidence"), list):
                raise ValueError(f"worker缺少结构化证据：{path}")
            workers.append(path_ref(path))
        value = read_jsonl(output) if step in {"literature", "candidates"} else read_json(output)
        validate_result(step, value, inputs)
        if step in {"literature", "candidates"}:
            atomic_write_json(directory / "provenance.json", {"inputs": refs, "workers": workers, "output": path_ref(output)})
        else:
            reserved = {"schema_version", "artifact_type", "run_id", "created_at", "producer", "inputs", "status"}
            domain = {key: item for key, item in value.items() if key not in reserved}
            kind = "research_context" if step == "research_intake" else "implementation_manifest" if "implementation" in step or step == "execution_repair" else step
            artifact_refs = [ref for ref in refs if Path(ref["path"]) != output] + workers
            if kind == "implementation_manifest":
                workspace = ensure_within(Path(domain["workspace"]), run_dir)
                cycle = output.parent
                if workspace != cycle / "workspace" or not workspace.is_dir():
                    raise ValueError("实现必须位于当前cycle的workspace")
                domain.update(snapshot_fields(workspace))
                # A dependency/environment repair may leave source bytes unchanged.
                # Always invalidate an earlier attempt, even within one second.
                domain["revision"] = previous_revision + 1
                # Bind only immutable scientific inputs. Repair diagnostics are
                # task inputs, not inputs of the replacement implementation.
                artifact_refs = [path_ref(path) for path in inputs if path.name in {"experiment_plan.json", "selection.json", "datasets.lock.json", "research_context.json", "preparation.md"}]
            status = "partial" if kind == "research_context" and domain.get("ready") is False else "complete"
            atomic_write_json(output, control_envelope(kind, run_dir.name, "kvresearch.research_agent", artifact_refs,
                                                       **domain, status=status))
            if status == "partial":
                return {"status": "needs_input", "missing_inputs": domain.get("missing_inputs", []), "output": str(output)}
        return {"status": "complete", "output": str(output), "task_packet": str(directory / "task.json")}
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        if output.is_file():
            output.replace(directory / f"invalid-output-{sha256_file(output)[:12]}{output.suffix}")
        return {"status": "invalid_output", "errors": [str(exc)], "task_packet": str(directory / "task.json")}
