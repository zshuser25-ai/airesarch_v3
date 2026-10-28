from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from kvresearch.codex_backend import MAIN_AGENT_MODEL, MAIN_AGENT_REASONING_EFFORT, MAIN_AGENT_SERVICE_TIER
from kvresearch.config import Config
from kvresearch.navigator import Action
from kvresearch.schema import ArtifactValidator, control_envelope
from kvresearch.util import atomic_write_json, atomic_write_text, path_ref, read_json, sha256_file, utcnow


JSONL_STEPS = {"literature": "literature", "candidates": "candidates"}
CONTROL_STEPS = {
    "evidence_synthesis": "evidence_synthesis",
    "problem_diagnosis": "problem_diagnosis",
    "selection": "selection",
    "experiment_plan": "experiment_plan",
    "implementation_manifest": "implementation_manifest",
    "implementation_repair": "implementation_manifest",
    "execution_repair": "implementation_manifest",
}

STAGE_WORKERS: dict[str, tuple[tuple[str, str], ...]] = {
    "literature": (
        ("extreme_kv_quantization", "Find verified work on 1-bit, binary, ternary, and extreme low-bit KV-cache quantization."),
        ("pq_residual_outliers", "Find verified work on product/vector quantization, codebooks, residuals, and outlier handling."),
        ("importance_allocation", "Find verified work on token/layer/head importance and Key/Value asymmetric allocation."),
        ("systems_accounting", "Find verified work on real storage, metadata, CUDA kernels, and long-context deployment costs."),
    ),
    "evidence_synthesis": (
        ("longbench_quality", "Synthesize mechanisms and failure modes most relevant to LongBench quality."),
        ("math_quality", "Synthesize mechanisms and failure modes most relevant to GSM8K and MATH500."),
        ("storage_feasibility", "Audit storage accounting, runtime feasibility, and implementation constraints."),
    ),
    "problem_diagnosis": (
        ("accuracy_diagnoser", "Diagnose task-level quality evidence and identify the strongest accuracy bottleneck."),
        ("quantization_diagnoser", "Diagnose Key/Value, layer/head/token, codebook, and residual error sources."),
        ("systems_diagnoser", "Diagnose actual storage, metadata, CUDA, memory, and execution constraints."),
    ),
    "candidates": (
        ("kv_bit_allocation_architecture", "Redesign joint Key/Value bit allocation across layers, heads, or token classes under the exact average one-bit budget."),
        ("residual_token_retention_architecture", "Redesign which tokens remain residual/full precision and how retained history is organized under the exact average one-bit budget."),
        ("representation_decoder_architecture", "Change the compressed representation or decoding/reconstruction architecture, not only codebook fitting or assignment distance."),
        ("stock_vecinfer_control", "Specify exact unmodified stock VecInfer with baseline codebooks and squared-Euclidean assignment; it is not selectable."),
    ),
    "selection": (
        ("scientific_reviewer", "Rank all candidates for causal plausibility, falsifiability, and joint quality upside."),
        ("storage_reviewer", "Rank all candidates for theoretical and actual storage compliance and control fairness."),
        ("engineering_reviewer", "Rank all candidates for VecInfer implementation risk, runtime cost, and debuggability."),
    ),
}

WORKER_AGENT_RUNTIME = {
    "context_isolation": "fork_turns_none",
    "model": MAIN_AGENT_MODEL,
    "reasoning_effort": MAIN_AGENT_REASONING_EFFORT,
    "service_tier": MAIN_AGENT_SERVICE_TIER,
}


_TRANSIENT_AGENT_FAILURE_MARKERS = (
    "selected model is at capacity",
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
    "rate limit",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "connection reset",
    "connection timed out",
    "server error",
)


def _is_transient_agent_failure(result: subprocess.CompletedProcess[str]) -> bool:
    """Return true only for retryable service/transport failures.

    Agent-generated schema or implementation failures are deliberately not
    retried here: those need the normal same-stage repair path and evidence.
    """
    if result.returncode == 0:
        return False
    message = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in message for marker in _TRANSIENT_AGENT_FAILURE_MARKERS)


def _run_agent_command(
    argv: list[str],
    cwd: Path,
    *,
    transient_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], list[subprocess.CompletedProcess[str]]]:
    """Run an agent, retrying bounded transient capacity/service failures."""
    attempts: list[subprocess.CompletedProcess[str]] = []
    retries = max(0, int(transient_retries))
    backoff = max(0.0, float(initial_backoff_seconds))
    max_backoff = max(0.0, float(max_backoff_seconds))
    for attempt_index in range(retries + 1):
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
        attempts.append(result)
        if not _is_transient_agent_failure(result) or attempt_index >= retries:
            return result, attempts
        if backoff:
            time.sleep(min(backoff, max_backoff))
        backoff = min(backoff * 2 if backoff else 0.0, max_backoff)
    raise AssertionError("unreachable")


def _worker_contract(task_id: str, worker_dir: Path, step: str) -> tuple[list[dict[str, str]], str]:
    assignments = []
    lines = [
        "\n# Required parallel subagent workflow",
        f"This stage has exactly {len(STAGE_WORKERS[step])} worker assignments.",
        "Before spawning, inspect the assigned worker paths and reuse only files that already satisfy the worker JSON contract.",
        "On a fresh task, spawn all assigned subagents concurrently. On recovery, spawn all missing or invalid assignments concurrently.",
        f"Every worker is a reasoning subagent and MUST be spawned with fork_turns='none', model={MAIN_AGENT_MODEL!r}, and reasoning_effort={MAIN_AGENT_REASONING_EFFORT!r}. Do not omit these arguments, inherit the parent context, or use a partial/full-history fork.",
        f"Every worker must use service_tier={MAIN_AGENT_SERVICE_TIER!r}, inherited from the host's Codex CLI configuration. Do not override the service tier or pass an unsupported service_tier argument to the spawn tool.",
        "Give every subagent the listed input paths and only its assigned objective. Wait for every subagent before synthesis.",
        "Each subagent must write only its own worker JSON path. The parent is the only agent allowed to write the formal output.",
        "Because fork_turns='none' supplies no parent conversation, put the assigned objective, listed disk input paths, exact output path, worker JSON contract, and necessary safety constraints directly in each spawn message.",
        "Every worker JSON must be one object with worker_id, role, status='complete', agent_runtime, findings (array), evidence (array), and recommendation.",
        "agent_runtime must be exactly " + json.dumps(WORKER_AGENT_RUNTIME, separators=(",", ":")) + "; the host rejects any other declaration.",
        "Do not replace missing worker evidence with parent speculation. If a worker fails, retry that bounded assignment once.",
    ]
    for index, (role, objective) in enumerate(STAGE_WORKERS[step]):
        worker_id = f"worker-{index:02d}"
        output = worker_dir / f"{worker_id}.json"
        assignments.append({"worker_id": worker_id, "role": role, "objective": objective, "output": str(output)})
        lines.append(f"- {worker_id} | role={role} | output={output} | objective={objective}")
    lines.extend([
        "After all worker files exist, read them, resolve conflicts, and write the single formal output.",
        "For literature and candidates JSONL, include worker_id or worker_ids on each derived row for auditability.",
    ])
    return assignments, "\n".join(lines)


def create_task_packet(config: Config, run_dir: Path, action: Action) -> tuple[Path, Path]:
    tasks_dir = run_dir / "agent_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    common = (prompt_dir / "common.md").read_text(encoding="utf-8")
    prompt_name = action.notes.removeprefix("prompt=") if action.notes.startswith("prompt=") else ""
    specific = Path(prompt_name).read_text(encoding="utf-8") if prompt_name and Path(prompt_name).is_file() else ""
    fingerprint_data = {
        "step": action.step, "output": action.output,
        "agent_runtime": {
            "model": MAIN_AGENT_MODEL,
            "reasoning_effort": MAIN_AGENT_REASONING_EFFORT,
            "service_tier": MAIN_AGENT_SERVICE_TIER,
        },
        "worker_agent_runtime": WORKER_AGENT_RUNTIME if action.step in STAGE_WORKERS else None,
        "inputs": [path_ref(Path(item)) if Path(item).is_file() else {"path": item, "sha256": None} for item in action.inputs],
        "prompt_contract_sha256": __import__("hashlib").sha256(
            (common + "\n" + specific + json.dumps(STAGE_WORKERS.get(action.step, ()), sort_keys=True)).encode("utf-8")
        ).hexdigest(),
    }
    fingerprint = __import__("hashlib").sha256(json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")).hexdigest()
    task_id = f"{action.step}-{fingerprint[:12]}"
    worker_dir = tasks_dir / task_id / "workers"
    assignments: list[dict[str, str]] = []
    parallel_contract = ""
    if action.step in STAGE_WORKERS:
        worker_dir.mkdir(parents=True, exist_ok=True)
        assignments, parallel_contract = _worker_contract(task_id, worker_dir, action.step)
    prompt = (
        common
        + "\n"
        + specific
        + parallel_contract
        + "\n\nInputs:\n"
        + "\n".join(f"- {item}" for item in action.inputs)
        + f"\n\nOutput: {action.output}\nArtifact type: {action.step}\n"
    )
    prompt_path = tasks_dir / f"{task_id}.md"
    packet_path = tasks_dir / f"{task_id}.json"
    atomic_write_text(prompt_path, prompt)
    atomic_write_json(
        packet_path,
        {
            "schema_version": 1,
            "task_id": task_id,
            "created_at": utcnow(),
            "step": action.step,
            "inputs": list(action.inputs),
            "output": action.output,
            "prompt": str(prompt_path),
            "parallel": bool(assignments),
            "worker_dir": str(worker_dir) if assignments else None,
            "workers": assignments,
            "input_fingerprint": fingerprint,
        },
    )
    return packet_path, prompt_path


def validate_worker_artifacts(packet_path: Path) -> tuple[bool, list[str], Path | None]:
    packet = read_json(packet_path)
    assignments = packet.get("workers", [])
    if not assignments:
        return True, [], None
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    expected_paths = {str(Path(item["output"]).resolve()) for item in assignments}
    actual_paths = {str(path.resolve()) for path in Path(packet["worker_dir"]).glob("worker-*.json")}
    if actual_paths != expected_paths:
        errors.append(f"worker file set mismatch: expected={sorted(expected_paths)} actual={sorted(actual_paths)}")
    for assignment in assignments:
        path = Path(assignment["output"])
        try:
            value = read_json(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        for field in ("worker_id", "role", "status", "agent_runtime", "findings", "evidence", "recommendation"):
            if field not in value:
                errors.append(f"{path}: missing {field}")
        if value.get("worker_id") != assignment["worker_id"] or value.get("role") != assignment["role"]:
            errors.append(f"{path}: worker identity mismatch")
        if value.get("status") != "complete":
            errors.append(f"{path}: status is not complete")
        if value.get("agent_runtime") != WORKER_AGENT_RUNTIME:
            errors.append(
                f"{path}: agent_runtime must declare isolated "
                f"{MAIN_AGENT_MODEL}/{MAIN_AGENT_REASONING_EFFORT}/{MAIN_AGENT_SERVICE_TIER} execution"
            )
        if not isinstance(value.get("findings"), list) or not isinstance(value.get("evidence"), list):
            errors.append(f"{path}: findings and evidence must be arrays")
        if not errors or path.is_file():
            records.append({"worker_id": assignment["worker_id"], "role": assignment["role"],
                            "agent_runtime": value.get("agent_runtime"),
                            "path": str(path), "sha256": sha256_file(path) if path.is_file() else None})
    if errors:
        return False, errors, None
    manifest_path = Path(packet["worker_dir"]).parent / "workers_manifest.json"
    atomic_write_json(manifest_path, {
        "schema_version": 1, "task_id": packet["task_id"], "step": packet["step"],
        "status": "complete", "worker_count": len(records), "workers": records,
    })
    return True, [], manifest_path


def run_agent_action(config: Config, run_dir: Path, action: Action) -> dict[str, Any]:
    packet, prompt = create_task_packet(config, run_dir, action)
    command_template = config.section("automation").get("agent_command", [])
    if not command_template:
        return {"status": "needs_agent_backend", "task_packet": str(packet), "prompt": str(prompt), "output": action.output}
    substitutions = {
        "task": str(packet), "prompt": str(prompt), "output": str(action.output), "run": str(run_dir),
        "max_parallel_subagents": str(config.section("automation").get("max_parallel_subagents", 4)),
    }
    argv = [str(token).format(**substitutions) for token in command_template]
    task_dir = packet.parent / packet.stem
    task_dir.mkdir(parents=True, exist_ok=True)
    automation = config.section("automation")
    result, attempts = _run_agent_command(
        argv,
        config.project_root,
        transient_retries=int(automation.get("agent_transient_retries", 6)),
        initial_backoff_seconds=float(automation.get("agent_retry_initial_seconds", 15)),
        max_backoff_seconds=float(automation.get("agent_retry_max_seconds", 60)),
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        atomic_write_text(task_dir / f"parent.attempt-{index:02d}.stdout.log", attempt.stdout)
        atomic_write_text(task_dir / f"parent.attempt-{index:02d}.stderr.log", attempt.stderr)
        stdout_parts.append(f"[agent attempt {index}/{len(attempts)}]\n{attempt.stdout}")
        stderr_parts.append(f"[agent attempt {index}/{len(attempts)}]\n{attempt.stderr}")
    combined_stdout = "\n".join(stdout_parts)
    combined_stderr = "\n".join(stderr_parts)
    atomic_write_text(task_dir / "parent.stdout.log", combined_stdout)
    atomic_write_text(task_dir / "parent.stderr.log", combined_stderr)
    if result.returncode != 0:
        return {
            "status": "agent_failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "attempts": len(attempts),
            "transient_failure": _is_transient_agent_failure(result),
            "task_packet": str(packet),
        }
    output = Path(action.output or "")
    workers_valid, worker_errors, worker_manifest = validate_worker_artifacts(packet)
    if not workers_valid:
        quarantined = None
        if output.is_file():
            quarantined = packet.parent / f"{packet.stem}.invalid-output{output.suffix}"
            output.replace(quarantined)
        return {"status": "invalid_workers", "errors": worker_errors, "task_packet": str(packet),
                "quarantined_output": str(quarantined) if quarantined else None}
    validator = ArtifactValidator(config.catalog_path)
    if action.step in JSONL_STEPS:
        validation = validator.validate_jsonl(output, JSONL_STEPS[action.step])
    else:
        if output.is_file():
            value = read_json(output)
            if action.step in {"implementation_manifest", "implementation_repair", "execution_repair"}:
                # These stages must finish through snapshot_implementation(),
                # whose host-generated manifest is already authoritative.  A
                # repair action reads and replaces the same path; wrapping it
                # with action.inputs would create a self-referential hash that
                # is guaranteed to drift immediately after the write.
                if not (
                    value.get("artifact_type") == "implementation_manifest"
                    and value.get("producer") == "kvresearch.workspace"
                ):
                    return {
                        "status": "invalid_output",
                        "errors": ["implementation stage did not use host snapshot_implementation"],
                        "attempts": len(attempts),
                        "task_packet": str(packet),
                        "output": str(output),
                    }
            else:
                # The host always owns the control envelope.  An Agent cannot
                # omit candidates.jsonl from selection provenance (which
                # would bypass architecture/stock-control semantic checks) or
                # substitute a different run/input identity.
                refs = [path_ref(Path(item)) for item in action.inputs if Path(item).is_file()]
                if worker_manifest:
                    refs.append(path_ref(worker_manifest))
                reserved = {"schema_version", "artifact_type", "run_id", "created_at", "producer", "inputs", "status"}
                domain_value = {key: item for key, item in value.items() if key not in reserved}
                wrapped = control_envelope(
                    CONTROL_STEPS[action.step],
                    run_dir.name,
                    "kvresearch.agent_backend",
                    refs,
                    **domain_value,
                )
                atomic_write_json(output, wrapped)
        validation = validator.validate_control(output, CONTROL_STEPS[action.step])
    return {"status": "complete" if validation.valid else "invalid_output", "errors": list(validation.errors),
            "attempts": len(attempts), "task_packet": str(packet), "output": str(output)}
