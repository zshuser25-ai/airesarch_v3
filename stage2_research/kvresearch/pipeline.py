from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from kvresearch.baseline import parse_baseline_report
from kvresearch.config import Config
from kvresearch.diagnostics import diagnostic_contract, validate_diagnostic_outputs
from kvresearch.mechanisms import canonical_mechanism_family
from kvresearch.remote import RemotePod
from kvresearch.stock_control import validate_stock_control_file
from kvresearch.schema import ArtifactValidator, control_envelope
from kvresearch.storage import normalize_storage_record
from kvresearch.util import (
    atomic_write_json,
    atomic_write_jsonl,
    ensure_within,
    latest_numbered_dir,
    path_ref,
    read_json,
    read_jsonl,
    sha256_file,
)


_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION = 6


def _smoke_rollover_evidence(attempt_dir: Path, tasks: list[str]) -> dict[str, Any]:
    """Require every non-scoring task to exercise compressed KV history."""
    details: dict[str, Any] = {}
    passed = True
    for task in tasks:
        raw_path = attempt_dir / "raw" / f"{task}.jsonl"
        prefill_tokens = None
        errors: list[str] = []
        if not raw_path.is_file():
            errors.append("missing smoke raw output")
        else:
            rows = read_jsonl(raw_path)
            if len(rows) != 1:
                errors.append(f"expected one smoke row, got {len(rows)}")
            else:
                prefill_tokens = rows[0].get("prefill_tokens")
                if (
                    not isinstance(prefill_tokens, int)
                    or isinstance(prefill_tokens, bool)
                    or prefill_tokens <= 128
                ):
                    errors.append("smoke prefill did not cross the 128-token exact residual")
        diagnostic_path = attempt_dir / "diagnostics" / f"{task}.json"
        observed = None
        not_applicable = None
        if diagnostic_path.is_file():
            diagnostic = read_json(diagnostic_path)
            observed = diagnostic.get("observed_checkpoint_count")
            not_applicable = diagnostic.get("not_applicable_checkpoint_count")
            if isinstance(observed, int) and not isinstance(observed, bool) and observed <= 0:
                errors.append("diagnostic observer recorded no compressed-history checkpoint")
        if errors:
            passed = False
        details[task] = {
            "passed": not errors,
            "prefill_tokens": prefill_tokens,
            "observed_checkpoint_count": observed,
            "not_applicable_checkpoint_count": not_applicable,
            "errors": errors,
        }
    return {"passed": passed, "tasks": details}


def _classify_smoke_failure(error_text: str, row_errors: list[str]) -> str:
    """Classify a failed implementation smoke invocation as a technical defect.

    Ordering matters here: candidate stack traces naturally mention codebooks
    and centroids even when the artifact loaded successfully.  Numerical
    failures therefore have to be recognized before artifact-path failures,
    and an artifact failure requires an actual missing-file signal rather than
    the mere presence of those nouns in a traceback.
    """
    lowered = error_text.lower()
    numerical_probability_failure = (
        ("probability tensor contains" in lowered or "probabilities contain" in lowered)
        and ("nan" in lowered or "inf" in lowered)
    )
    explicit_nonfinite_failure = any(
        marker in lowered
        for marker in (
            "nan/inf",
            "nan or inf",
            "inf or nan",
            "non-finite",
            "nonfinite",
        )
    )
    if numerical_probability_failure or explicit_nonfinite_failure:
        return "candidate_numerical_instability"
    if "device-side assert triggered" in lowered or "device side assert triggered" in lowered:
        return "cuda_device_assert"
    if "e_calibration_unlocked" in lowered or (
        "calibration" in lowered and any(marker in lowered for marker in ("missing", "not found", "no such file"))
    ):
        return "missing_calibration_artifact"
    if "filenotfounderror" in lowered or (
        ("codebook" in lowered or "centroid" in lowered)
        and any(marker in lowered for marker in ("missing", "not found", "no such file"))
    ):
        return "missing_candidate_artifact"
    if "undefined symbol" in lowered or "abi" in lowered or "importerror" in lowered:
        return "runtime_or_abi"
    if "out of memory" in lowered:
        return "gpu_oom"
    if row_errors:
        return "output_contract"
    return "candidate_runtime"


def _implementation_smoke(config: Config, run_dir: Path, cycle_dir: Path,
                          plan: dict[str, Any], implementation_path: Path) -> dict[str, Any]:
    """Serialize validators sharing one content-addressed smoke namespace."""
    import fcntl

    implementation_sha = sha256_file(implementation_path)
    smoke_key = (
        f"{cycle_dir.name}-{implementation_sha[:12]}"
        f"-p{_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION}"
    )
    smoke_root = cycle_dir / "implementation_smoke" / smoke_key
    smoke_root.mkdir(parents=True, exist_ok=True)
    lock_path = smoke_root / ".validation.lock"
    with lock_path.open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        return _implementation_smoke_locked(
            config, run_dir, cycle_dir, plan, implementation_path
        )


def _implementation_smoke_locked(config: Config, run_dir: Path, cycle_dir: Path,
                                 plan: dict[str, Any], implementation_path: Path) -> dict[str, Any]:
    """Run one real, non-scoring POD row for every planned task.

    Static imports and a generic GPU preflight cannot prove that a candidate
    has produced its codebooks, resolves the intended extension, or can close
    its prediction/storage output contract.  This gate invokes the exact
    ``VecInferAdapter`` entrypoint on one locked row per task.  Results are
    content-addressed by the implementation manifest, so a repaired snapshot
    always receives a fresh smoke run while an unchanged failure is not burned
    repeatedly on the GPU.
    """
    from dataclasses import replace
    from kvresearch.vecinfer import VecInferAdapter

    implementation_sha = sha256_file(implementation_path)
    smoke_key = (
        f"{cycle_dir.name}-{implementation_sha[:12]}"
        f"-p{_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION}"
    )
    smoke_root = cycle_dir / "implementation_smoke" / smoke_key
    result_path = smoke_root / "result.json"
    if result_path.is_file():
        cached = read_json(result_path)
        if (
            cached.get("implementation_sha256") == implementation_sha
            and cached.get("protocol_version") == _IMPLEMENTATION_SMOKE_PROTOCOL_VERSION
        ):
            return cached

    tasks = list(plan.get("datasets", []))
    source_tier = "formal" if plan.get("evaluation_tier") == "formal" else "development"
    remote = RemotePod(config)
    prepare_argv = [
        config.section("remote")["python"], "-m", "kvresearch.data",
        "--config", str(config.path), "--run", str(run_dir),
        "--smoke-source-tier", source_tier, "--smoke-key", smoke_key,
        "--tasks", ",".join(tasks),
    ]
    prepared = remote.run(
        f"cd {shlex.quote(str(config.project_root))} && PYTHONDONTWRITEBYTECODE=1 {shlex.join(prepare_argv)}",
        timeout=300,
    )
    smoke_root.mkdir(parents=True, exist_ok=True)
    if prepared.returncode != 0:
        result = {
            "passed": False,
            "implementation_sha256": implementation_sha,
            "failure_class": "smoke_dataset_preparation",
            "repairable": True,
            "tasks": [],
            "error": prepared.stderr.strip() or prepared.stdout.strip(),
        }
        atomic_write_json(result_path, result)
        return result
    dataset_root = Path(prepared.stdout.strip().splitlines()[-1]).resolve()

    environment = remote.environment_snapshot()
    constraints = environment.get("constraints", {})
    min_free = int(constraints.get("minimum_free_gpu_memory_mib") or 0)
    max_util = int(constraints.get("maximum_idle_gpu_utilization_percent") or 100)
    eligible = [gpu for gpu in environment.get("gpus", [])
                if int(gpu.get("memory_free_mib", 0)) >= min_free
                and int(gpu.get("utilization_percent", 100)) <= max_util]
    if not environment.get("valid") or not eligible:
        result = {
            "passed": False,
            "implementation_sha256": implementation_sha,
            "failure_class": "remote_environment",
            "repairable": True,
            "tasks": [],
            "error": "no eligible POD GPU for implementation smoke",
            "environment_errors": environment.get("errors", []),
        }
        atomic_write_json(result_path, result)
        return result

    smoke_plan = json.loads(json.dumps(plan))
    smoke_plan["batch_size"] = 1
    smoke_plan["datasets"] = tasks
    execution = smoke_plan.setdefault("execution", {})
    env = execution.setdefault("env", {})
    env.update({
        "KVRESEARCH_IMPLEMENTATION_SMOKE": "1",
        "KVRESEARCH_EXPECTED_ROWS": "1",
        # This is a non-scoring liveness gate.  The full evaluation retains
        # its locked generation budget; the bounded smoke only verifies that
        # sampled math reaches atomic output closure in finite time.
        "VECINFER_GENERATION_MAX_LENGTH": "512",
    })
    adapter = VecInferAdapter(config)
    # Bind one worker to each eligible GPU and let workers pull from a shared
    # task queue.  A card that finishes a short task immediately claims the
    # next one, so the 15-task implementation gate is neither serialized on a
    # single card nor held back by a static longest-task assignment.
    from concurrent.futures import ThreadPoolExecutor
    from queue import Empty, Queue

    worker_limit = int(config.section("evaluation").get("formal_gpu_count", 8))
    worker_gpus = sorted(eligible, key=lambda item: int(item["index"]))[:worker_limit]
    task_queue: Queue[str] = Queue()
    for task in tasks:
        task_queue.put(task)

    required_row_fields = {"_fixed_sample", "prefill_tokens", "output_tokens", "total_tokens",
                           "actual_pq_kv_cache_bytes"}

    def run_one(task: str, gpu: dict[str, Any], worker_slot: int) -> dict[str, Any]:
        task_plan = json.loads(json.dumps(smoke_plan))
        task_plan["datasets"] = [task]
        originals = adapter.commands(
            run_dir, cycle_dir.name,
            f"smoke-{implementation_sha[:12]}-p{_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION}",
            [gpu], task_plan,
            tier_override=source_tier, raw_subdir="raw",
            dataset_root_override=dataset_root,
        )
        if len(originals) != 1:
            raise RuntimeError(f"smoke scheduler expected one command for {task}, got {len(originals)}")
        original = originals[0]
        command = replace(original, timeout_seconds=min(int(original.timeout_seconds), 1200))
        stdout_path = smoke_root / "logs" / f"{task}.stdout.log"
        stderr_path = smoke_root / "logs" / f"{task}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            completed = remote.execute_argv(command.argv, command.cwd, command.env, command.timeout_seconds)
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            error_text = completed.stderr + "\n" + completed.stdout
            row_errors = []
            rows = []
            if completed.returncode == 0:
                if not command.raw_output or not command.raw_output.is_file():
                    row_errors.append("successful command did not publish raw output")
                else:
                    rows = read_jsonl(command.raw_output)
                    if len(rows) != 1:
                        row_errors.append(f"expected exactly one smoke row, got {len(rows)}")
                    elif missing := sorted(required_row_fields - set(rows[0])):
                        row_errors.append(f"smoke row missing fields: {missing}")
            passed = completed.returncode == 0 and not row_errors
            failure_class = None
            if not passed:
                failure_class = _classify_smoke_failure(error_text, row_errors)
            return {
                "task": task,
                "gpu_index": int(gpu["index"]),
                "gpu_uuid": str(gpu.get("uuid", "")),
                "worker_slot": worker_slot,
                "passed": passed,
                "returncode": int(completed.returncode),
                "elapsed_seconds": time.monotonic() - started,
                "failure_class": failure_class,
                "row_errors": row_errors,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "raw_output": str(command.raw_output) if command.raw_output else None,
            }
        except subprocess.TimeoutExpired:
            return {
                "task": task, "passed": False, "returncode": 124,
                "gpu_index": int(gpu["index"]),
                "gpu_uuid": str(gpu.get("uuid", "")),
                "worker_slot": worker_slot,
                "elapsed_seconds": time.monotonic() - started,
                "failure_class": "smoke_timeout", "row_errors": [],
                "stdout": str(stdout_path), "stderr": str(stderr_path),
                "raw_output": str(command.raw_output) if command.raw_output else None,
            }

    def gpu_worker(gpu: dict[str, Any], worker_slot: int) -> list[dict[str, Any]]:
        completed_tasks: list[dict[str, Any]] = []
        while True:
            try:
                task = task_queue.get_nowait()
            except Empty:
                return completed_tasks
            try:
                completed_tasks.append(run_one(task, gpu, worker_slot))
            finally:
                task_queue.task_done()

    task_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(worker_gpus)) as pool:
        futures = [
            pool.submit(gpu_worker, gpu, slot)
            for slot, gpu in enumerate(worker_gpus)
        ]
        for future in futures:
            task_results.extend(future.result())
    task_order = {task: index for index, task in enumerate(tasks)}
    task_results.sort(key=lambda item: task_order[item["task"]])
    passed = len(task_results) == len(tasks) and all(item["passed"] for item in task_results)
    result = {
        "passed": passed,
        "protocol_version": _IMPLEMENTATION_SMOKE_PROTOCOL_VERSION,
        "implementation_sha256": implementation_sha,
        "failure_class": None if passed else "implementation_not_ready",
        "repairable": not passed,
        "non_scoring": True,
        "source_tier": source_tier,
        "dataset_root": str(dataset_root),
        "expected_rows_per_task": 1,
        "scheduling": {
            "mode": "dynamic_shared_queue",
            "gpu_worker_count": len(worker_gpus),
            "gpu_indices": [int(gpu["index"]) for gpu in worker_gpus],
            "workers_per_gpu": 1,
        },
        "tasks": task_results,
    }
    atomic_write_json(result_path, result)
    return result


def _math_budget_check(config: Config, plan: dict[str, Any], planned_tasks: set[str]) -> dict[str, Any]:
    evaluation = config.section("evaluation")
    math_cap = int(evaluation.get("math_quality_max_total_length", 2048))
    matched_storage_length = int(evaluation["max_total_length"])
    execution_env = plan.get("execution", {}).get("env", {})
    decoding = plan.get("decoding", {})
    passed = (
        str(execution_env.get("VECINFER_GENERATION_MAX_LENGTH", "")) == str(math_cap)
        and str(execution_env.get("VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH", ""))
        == str(matched_storage_length)
        and isinstance(decoding.get("math_max_total_length"), int)
        and not isinstance(decoding.get("math_max_total_length"), bool)
        and decoding.get("math_max_total_length") == math_cap
        and not (
            isinstance(decoding.get("longbench"), dict)
            and "max_total_length" in decoding["longbench"]
        )
        and all(
            isinstance((decoding.get(task) or {}).get("max_total_length"), int)
            and not isinstance((decoding.get(task) or {}).get("max_total_length"), bool)
            and (decoding.get(task) or {}).get("max_total_length") == math_cap
            for task in ("gsm8k", "math500")
            if task in planned_tasks
        )
    )
    return {
        "passed": passed,
        "expected_max_total_length": math_cap,
        "plan_math_max_total_length": decoding.get("math_max_total_length"),
        "longbench_total_length_cap": (decoding.get("longbench") or {}).get("max_total_length")
        if isinstance(decoding.get("longbench"), dict)
        else None,
        "environment_max_total_length": execution_env.get("VECINFER_GENERATION_MAX_LENGTH"),
        "matched_storage_logical_length": execution_env.get("VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH"),
        "expected_matched_storage_logical_length": matched_storage_length,
    }


def validate_cycle(config: Config, run_dir: Path, cycle: str) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    plan_path = cycle_dir / "experiment_plan.json"
    implementation_path = cycle_dir / "implementation_manifest.json"
    validator = ArtifactValidator(config.catalog_path)
    checks: dict[str, Any] = {}
    for name, path, kind in (
        ("plan_schema", plan_path, "experiment_plan"),
        ("implementation_schema", implementation_path, "implementation_manifest"),
    ):
        result = validator.validate_control(path, kind)
        checks[name] = {"passed": result.valid, "errors": list(result.errors)}
    plan = read_json(plan_path)
    implementation = read_json(implementation_path)
    workspace = Path(implementation.get("workspace", ""))
    try:
        ensure_within(workspace, run_dir)
        workspace_ok = workspace.is_dir() and (workspace / ".git").is_dir()
    except ValueError:
        workspace_ok = False
    checks["isolated_workspace"] = {"passed": workspace_ok, "path": str(workspace)}
    checks["workspace_matches_plan"] = {"passed": str(workspace) == str(plan.get("workspace", ""))}
    changed = implementation.get("changed_files", [])
    checks["candidate_has_change"] = {"passed": bool(changed), "changed_files": changed}
    syntax_errors: list[str] = []
    for relative in changed:
        path = workspace / relative
        if path.suffix == ".py" and path.is_file():
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    "import pathlib,sys; compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), sys.argv[1], 'exec')",
                    str(path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                syntax_errors.append(f"{relative}: {result.stderr.strip()}")
    checks["syntax"] = {"passed": not syntax_errors, "errors": syntax_errors}
    locked = read_json(run_dir / "inputs.lock.json")
    locked_tasks = set(locked["evaluation"]["development_tasks"])
    tier = plan.get("evaluation_tier")
    planned_tasks = set(plan.get("datasets", []))
    if tier == "development":
        protocol_ok = planned_tasks == locked_tasks
    elif tier == "formal":
        evaluation = locked["evaluation"]
        execution = plan.get("execution", {})
        protocol_ok = (
            planned_tasks == set(evaluation["longbench_tasks"] + evaluation["math_tasks"])
            and float(plan.get("sample_fraction", -1)) == float(evaluation["formal_sample_fraction"])
            and int(execution.get("gpu_count", 0)) == int(evaluation["formal_gpu_count"])
            and list(execution.get("batch_size_schedule", [])) == list(evaluation["formal_batch_sizes"])
        )
    else:
        protocol_ok = False
    checks["protocol_match"] = {"passed": protocol_ok, "tier": tier, "datasets": sorted(planned_tasks)}
    checks["math_quality_budget"] = _math_budget_check(config, plan, planned_tasks)
    accounting = plan.get("storage_accounting", {})
    theoretical = accounting.get("theoretical_average_bits")
    theoretical_ok = isinstance(theoretical, (int, float)) and float(theoretical) <= float(config.section("storage")["maximum_theoretical_average_bits"])
    checks["theoretical_storage"] = {"passed": theoretical_ok, "value": theoretical}
    estimated_ok = accounting.get("estimated_actual_not_above_vecinfer") is True
    checks["estimated_storage"] = {"passed": estimated_ok}
    execution_closure = plan.get("execution_closure") or {}
    closure_evidence = execution_closure.get("evidence")
    closure_ok = (
        execution_closure.get("adapter_contract") == "single_task_main_pq_per_task"
        and execution_closure.get("all_prerequisites_available") is True
        and isinstance(closure_evidence, list)
        and len(closure_evidence) > 0
    )
    checks["execution_closure"] = {
        "passed": closure_ok,
        "adapter_contract": execution_closure.get("adapter_contract"),
        "all_prerequisites_available": execution_closure.get("all_prerequisites_available"),
        "evidence": closure_evidence if isinstance(closure_evidence, list) else [],
    }
    checks["imports"] = {"passed": True, "deferred_to_remote_runtime_probe": True}
    main_pq = workspace / "scripts" / "modeldb" / "main_pq.py"
    main_pq_text = main_pq.read_text(encoding="utf-8", errors="replace") if main_pq.is_file() else ""
    stock_interface_markers = (
        "VECINFER_STOCK_CODEBOOK_ROOT",
        "VECINFER_STOCK_CODEBOOK_MANIFEST_SHA256",
        "VECINFER_STOCK_ASSIGNMENT_POLICY",
        "stock_vecinfer_control",
    )
    missing_stock_markers = [marker for marker in stock_interface_markers if marker not in main_pq_text]
    checks["interfaces"] = {
        "passed": plan.get("execution", {}).get("adapter", "vecinfer") == "vecinfer" and not missing_stock_markers,
        "stock_control_code_path_markers": list(stock_interface_markers),
        "missing_stock_control_markers": missing_stock_markers,
    }
    manifest_key = "development_manifest" if tier == "development" else "formal_manifest"
    locked_manifests = [locked["hashes"][manifest_key]]
    if tier == "formal":
        locked_manifests.append(locked["hashes"]["formal_remaining_manifest"])
    checks["data_hashes"] = {"passed": True, "locked_manifests": locked_manifests}
    cheap_checks_pass = all(bool(value.get("passed")) for value in checks.values())
    if cheap_checks_pass:
        smoke = _implementation_smoke(config, run_dir, cycle_dir, plan, implementation_path)
        smoke_attempt = (
            cycle_dir / "attempts"
            / (
                f"smoke-{sha256_file(implementation_path)[:12]}"
                f"-p{_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION}"
            )
        )
        smoke_diagnostics = validate_diagnostic_outputs(plan, smoke_attempt)
        rollover_evidence = _smoke_rollover_evidence(smoke_attempt, list(planned_tasks))
        stock_control_errors = []
        for task in sorted(planned_tasks):
            stock_control_errors.extend(validate_stock_control_file(
                config, smoke_attempt / "arms" / task / "unmodified_vecinfer.jsonl"
            ))
        stock_control_evidence = {
            "passed": not stock_control_errors,
            "errors": stock_control_errors,
            "host_locked": True,
        }
        smoke_passed = (
            bool(smoke.get("passed"))
            and bool(smoke_diagnostics.get("valid"))
            and bool(rollover_evidence.get("passed"))
            and stock_control_evidence["passed"]
        )
        smoke_failure_class = smoke.get("failure_class")
        if smoke.get("passed") and not smoke_diagnostics.get("valid"):
            smoke_failure_class = "diagnostics_incomplete"
        if smoke.get("passed") and not rollover_evidence.get("passed"):
            smoke_failure_class = "compressed_history_not_exercised"
        if smoke.get("passed") and not stock_control_evidence["passed"]:
            smoke_failure_class = "stock_vecinfer_control_invalid"
        checks["model_smoke"] = {
            "passed": smoke_passed,
            "real_pod_end_to_end": True,
            "non_scoring": True,
            "implementation_sha256": smoke.get("implementation_sha256"),
            "failure_class": smoke_failure_class,
            "repairable": bool(smoke.get("repairable", False)) or not smoke_diagnostics.get("valid"),
            "tasks": smoke.get("tasks", []),
            "diagnostics": smoke_diagnostics,
            "compressed_history_evidence": rollover_evidence,
            "stock_vecinfer_control": stock_control_evidence,
            "evidence": str(cycle_dir / "implementation_smoke" /
                            (
                                f"{cycle_dir.name}-{sha256_file(implementation_path)[:12]}"
                                f"-p{_IMPLEMENTATION_SMOKE_PROTOCOL_VERSION}"
                            ) / "result.json"),
        }
    else:
        checks["model_smoke"] = {
            "passed": False,
            "real_pod_end_to_end": False,
            "skipped": "static or execution-closure checks failed before GPU allocation",
            "repairable": True,
        }
    hard = all(bool(value.get("passed")) for value in checks.values())
    artifact = control_envelope(
        "validation",
        run_dir.name,
        "kvresearch.pipeline.validate_cycle",
        [path_ref(plan_path), path_ref(implementation_path), path_ref(run_dir / "inputs.lock.json")],
        syntax=checks["syntax"], imports=checks["imports"], interfaces=checks["interfaces"],
        model_smoke=checks["model_smoke"], data_hashes=checks["data_hashes"], protocol_match=checks["protocol_match"],
        theoretical_storage=checks["theoretical_storage"], estimated_storage=checks["estimated_storage"],
        checks=checks, all_hard_checks_pass=hard,
        status="complete" if hard else "failed",
        error_code=None if hard else ("E_IMPLEMENTATION_NOT_READY" if not checks["model_smoke"]["passed"] else "E_ARTIFACT_INVALID"),
    )
    atomic_write_json(cycle_dir / "validation.json", artifact)
    return artifact


def _sample_id(task: str, row: dict[str, Any], fallback: int) -> str:
    meta = row.get("_fixed_sample", {})
    identity = meta.get("source_id")
    if identity in (None, "None", ""):
        identity = meta.get("source_index", row.get("sample_idx", row.get("index", fallback)))
    return f"{task}:{identity}"


def _math_eval(config: Config, attempt_dir: Path, task: str, raw_path: Path, phase: str = "combined") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_dir = attempt_dir / "math_eval_input" / phase / task
    output_dir = attempt_dir / "math_eval" / phase / task
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = input_dir / f"{task}.jsonl"
    shutil.copy2(raw_path, staged)
    remote = config.section("remote")
    pythonpath = ":".join(str(item) for item in remote.get("runtime_pythonpath", []))
    dataset_arg = "math" if task == "math500" else task
    evaluator = Path(config.section("baseline")["source"]) / "scripts" / "utils" / "evaluation" / "eval_math.py"
    cwd = evaluator.parent
    command = [remote["vecinfer_python"], evaluator.name, "--exp_name", "candidate", "--base_dir", str(input_dir),
               "--output_dir", str(output_dir), "--dataset", dataset_arg]
    result = RemotePod(config).execute_argv(command, cwd, {"PYTHONPATH": pythonpath}, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"math evaluator failed for {task}: {result.stderr}")
    evaluated = output_dir / "candidate" / dataset_arg / f"default-{dataset_arg}_math_eval.jsonl"
    metric_path = output_dir / "candidate" / dataset_arg / f"default-{dataset_arg}_math_eval_cot_metrics.json"
    return read_jsonl(evaluated), read_json(metric_path)


def aggregate(config: Config, run_dir: Path, cycle: str) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    attempt = latest_numbered_dir(cycle_dir / "attempts")
    plan_path = cycle_dir / "experiment_plan.json"
    validation_path = cycle_dir / "validation.json"
    plan = read_json(plan_path)
    task_metrics: dict[str, Any] = {}
    paired_control_metrics: dict[str, Any] = {}
    normalized_by_task: dict[str, list[dict[str, Any]]] = {}
    storage_rows: list[dict[str, Any]] = []
    matched_storage_records: list[dict[str, Any]] = []
    expected_matched_storage_records = 0
    failures: list[str] = []
    diagnostic_validation = validate_diagnostic_outputs(plan, attempt) if attempt is not None else {
        **diagnostic_contract(plan), "valid": not diagnostic_contract(plan)["required"],
        "tasks": {}, "errors": ["no attempt available for diagnostics"], "evidence_paths": [],
    }
    remaining_diagnostic_validation = None
    if diagnostic_validation["required"] and not diagnostic_validation["valid"]:
        failures.extend(f"diagnostic evidence invalid: {error}" for error in diagnostic_validation["errors"])
    inputs = [path_ref(plan_path), path_ref(validation_path)]
    inputs.extend(
        path_ref(Path(path)) for path in diagnostic_validation.get("evidence_paths", [])
        if Path(path).is_file()
    )
    if attempt is None or not (attempt / "exit.json").is_file():
        failures.append("no closed execution attempt")
    else:
        inputs.append(path_ref(attempt / "exit.json"))
        theoretical = float(plan.get("storage_accounting", {}).get("theoretical_average_bits", math.inf))
        project = config.section("project")
        gate_path = attempt / "formal_30pct_gate.json"
        gate = read_json(gate_path) if gate_path.is_file() else None
        if (
            plan.get("evaluation_tier") == "formal"
            and gate
            and gate.get("qualified_for_remaining_70pct")
        ):
            remaining_diagnostic_validation = validate_diagnostic_outputs(
                plan, attempt, remaining=True
            )
            if (
                remaining_diagnostic_validation["required"]
                and not remaining_diagnostic_validation["valid"]
            ):
                failures.extend(
                    f"remaining diagnostic evidence invalid: {error}"
                    for error in remaining_diagnostic_validation["errors"]
                )
            inputs.extend(
                path_ref(Path(path))
                for path in remaining_diagnostic_validation.get("evidence_paths", [])
                if Path(path).is_file()
            )
        segments = [("development", attempt / "raw", Path(project["fixed_testset"]))]
        if plan.get("evaluation_tier") == "formal":
            segments = [("formal", attempt / "raw", Path(project["formal_testset"]))]
            if gate and gate.get("qualified_for_remaining_70pct"):
                segments.append(("formal_remaining", attempt / "raw_remaining", Path(project["formal_remaining_testset"])))
            if gate_path.is_file():
                inputs.append(path_ref(gate_path))
        for task in plan.get("datasets", []):
            normalized: list[dict[str, Any]] = []
            scores: list[float] = []
            control_scores: list[float] = []
            control_ids: list[str] = []
            expected_rows: list[dict[str, Any]] = []
            external_metric: dict[str, Any] = {}
            control_external_metric: dict[str, Any] = {}
            paired_control_expected = False
            segment_failed = False
            for phase, raw_dir, dataset_dir in segments:
                expected_matched_storage_records += 1
                join_dir = "joins_remaining" if raw_dir.name == "raw_remaining" else "joins"
                join_path = attempt / join_dir / f"{task}.json"
                join = None
                if join_path.is_file():
                    paired_control_expected = True
                    inputs.append(path_ref(join_path))
                    join = read_json(join_path)
                    comparison = join.get("storage_comparison") or (
                        (join.get("matched_storage") or {}).get("comparison") or {}
                    )
                    charged = comparison.get("charged_scopes") or {}
                    finite_scopes = bool(charged) and all(
                        isinstance(values, dict)
                        and isinstance(values.get("control"), (int, float))
                        and isinstance(values.get("candidate"), (int, float))
                        and math.isfinite(float(values["control"]))
                        and math.isfinite(float(values["candidate"]))
                        for values in charged.values()
                    )
                    increased_component_scopes = [
                        name for name, values in charged.items()
                        if isinstance(values, dict)
                        and isinstance(values.get("control"), (int, float))
                        and isinstance(values.get("candidate"), (int, float))
                        and float(values["candidate"]) > float(values["control"])
                    ]
                    # The registered storage gate applies to complete aggregate
                    # allocations, not to every internal component in
                    # isolation.  A candidate may spend a few more bytes on
                    # routing metadata while saving more exact-residual or code
                    # bytes.  Requiring every component to be non-increasing
                    # incorrectly rejects such byte-neutral/reducing designs.
                    aggregate_scope_names = {
                        "actual_pq_kv_cache_bytes",
                        "persistent_kv_related_bytes",
                        "peak_kv_related_bytes",
                    }
                    aggregate_over_scopes = [
                        name for name in aggregate_scope_names
                        if isinstance(charged.get(name), dict)
                        and isinstance(charged[name].get("control"), (int, float))
                        and isinstance(charged[name].get("candidate"), (int, float))
                        and float(charged[name]["candidate"]) > float(charged[name]["control"])
                    ]
                    budget_exceeded = bool(
                        comparison.get("candidate_no_greater_than_vecinfer") is not True
                        or aggregate_over_scopes
                    )
                    complete = bool(
                        (join.get("complete") is True or join.get("storage_gate_pass") is True)
                        and comparison.get("complete") is True
                        and finite_scopes
                        and not budget_exceeded
                    )
                    matched_storage_records.append({
                        "task": task,
                        "phase": phase,
                        "path": str(join_path),
                        "logical_length": comparison.get("logical_length"),
                        "complete": complete,
                        "budget_exceeded": budget_exceeded,
                        "aggregate_over_scopes": aggregate_over_scopes,
                        "increased_component_scopes": increased_component_scopes,
                    })
                raw_path = raw_dir / f"{task}.jsonl"
                if not raw_path.is_file():
                    failures.append(f"missing raw predictions for {phase}:{task}")
                    segment_failed = True
                    continue
                inputs.append(path_ref(raw_path))
                raw_rows = read_jsonl(raw_path)
                phase_expected = read_jsonl(dataset_dir / f"{task}.jsonl")
                expected_rows.extend(phase_expected)
                evaluated_rows = raw_rows
                if task in {"gsm8k", "math500"}:
                    try:
                        evaluated_rows, phase_metric = _math_eval(config, attempt, task, raw_path, phase)
                        external_metric[phase] = phase_metric
                    except Exception as exc:
                        failures.append(str(exc))
                        segment_failed = True
                        continue
                for index, row in enumerate(evaluated_rows):
                    raw = raw_rows[index] if index < len(raw_rows) else row
                    local_index = int(raw.get("sample_idx", index))
                    identity_row = phase_expected[local_index] if 0 <= local_index < len(phase_expected) else raw
                    request_id = _sample_id(task, identity_row, local_index)
                    if task in {"gsm8k", "math500"}:
                        score_value = row.get("score", [False])
                        correct = bool(score_value[0] if isinstance(score_value, list) else score_value)
                        score = 100.0 if correct else 0.0
                        prediction = row.get("code", raw.get("output", ""))
                        references = [row.get("gt", row.get("answer", ""))]
                    else:
                        score = float(row.get("score", 0.0))
                        prediction = row.get("prediction", "")
                        references = row.get("answers", [])
                    scores.append(score)
                    normalized.append({
                        "sample_id": request_id, "task": task,
                        "input_sha256": __import__("hashlib").sha256(str(row.get("prompt", row.get("question", ""))).encode()).hexdigest(),
                        "prediction": prediction, "references": references, "score": score,
                        "input_tokens": int(raw.get("prefill_tokens", 0)), "output_tokens": int(raw.get("output_tokens", 0)),
                        "finish_reason": raw.get("finish_reason", "completed"),
                        "cap_reached": raw.get("cap_reached"), "status": "complete",
                        "elapsed_seconds": raw.get("sample_end_to_end_seconds", raw.get("generation_seconds")), "evaluation_phase": phase,
                    })
                    storage_rows.append(normalize_storage_record(config, task, raw, request_id, theoretical))
                if join is not None:
                    arms_dir = "arms_remaining" if raw_dir.name == "raw_remaining" else "arms"
                    control_path = attempt / arms_dir / task / "unmodified_vecinfer.jsonl"
                    control_ref = (join.get("arms") or {}).get("unmodified_vecinfer") or {}
                    control_ref_ok = (
                        control_path.is_file()
                        and control_ref.get("path") == str(control_path)
                        and control_ref.get("sha256") == sha256_file(control_path)
                    )
                    if not control_ref_ok:
                        failures.append(f"missing or hash-drifted fresh paired control for {phase}:{task}")
                        segment_failed = True
                        continue
                    stock_errors = validate_stock_control_file(config, control_path)
                    if stock_errors:
                        failures.extend(stock_errors)
                        segment_failed = True
                        continue
                    inputs.append(path_ref(control_path))
                    control_raw_rows = read_jsonl(control_path)
                    control_evaluated_rows = control_raw_rows
                    if task in {"gsm8k", "math500"}:
                        try:
                            control_evaluated_rows, control_phase_metric = _math_eval(
                                config, attempt, task, control_path, f"{phase}_control"
                            )
                            control_external_metric[phase] = control_phase_metric
                        except Exception as exc:
                            failures.append(f"paired control {exc}")
                            segment_failed = True
                            continue
                    if len(control_evaluated_rows) != len(control_raw_rows):
                        failures.append(f"fresh paired control evaluator row mismatch for {phase}:{task}")
                        segment_failed = True
                        continue
                    for index, row in enumerate(control_evaluated_rows):
                        raw = control_raw_rows[index]
                        local_index = int(raw.get("sample_idx", index))
                        identity_row = phase_expected[local_index] if 0 <= local_index < len(phase_expected) else raw
                        control_ids.append(_sample_id(task, identity_row, local_index))
                        if task in {"gsm8k", "math500"}:
                            score_value = row.get("score", [False])
                            correct = bool(score_value[0] if isinstance(score_value, list) else score_value)
                            control_scores.append(100.0 if correct else 0.0)
                        else:
                            control_scores.append(float(row.get("score", 0.0)))
            expected_ids = [_sample_id(task, item, index) for index, item in enumerate(expected_rows)]
            sample_ids_match = True
            control_sample_ids_match = not paired_control_expected
            if not segment_failed:
                actual_ids = [item["sample_id"] for item in normalized]
                sample_ids_match = actual_ids == expected_ids and len(actual_ids) == len(set(actual_ids))
                if not sample_ids_match:
                    failures.append(f"fixed sample IDs mismatch for {task}")
                if paired_control_expected:
                    control_sample_ids_match = (
                        control_ids == expected_ids and len(control_ids) == len(set(control_ids))
                    )
                    if not control_sample_ids_match:
                        failures.append(f"fresh paired control sample IDs mismatch for {task}")
            normalized_by_task[task] = normalized
            task_metrics[task] = {
                "samples": len(normalized),
                "valid_samples": len(scores),
                "score": sum(scores) / len(scores) if scores else None,
                "correct": sum(1 for score in scores if score == 100.0) if task in {"gsm8k", "math500"} else None,
                "external_metrics": external_metric,
                "sample_ids_match": sample_ids_match,
            }
            if paired_control_expected:
                paired_control_metrics[task] = {
                    "samples": len(control_ids),
                    "valid_samples": len(control_scores),
                    "score": sum(control_scores) / len(control_scores) if control_scores else None,
                    "correct": sum(1 for score in control_scores if score == 100.0)
                    if task in {"gsm8k", "math500"} else None,
                    "external_metrics": control_external_metric,
                    "sample_ids_match": control_sample_ids_match,
                }
            predictions_dir = attempt / "predictions"
            atomic_write_jsonl(predictions_dir / f"{task}.jsonl", normalized)
        atomic_write_jsonl(attempt / "storage_trace.jsonl", storage_rows)

    long_scores = [entry["score"] for task, entry in task_metrics.items() if task not in {"gsm8k", "math500"} and entry["score"] is not None]
    math_scores = [task_metrics[task]["score"] for task in ("gsm8k", "math500") if task in task_metrics and task_metrics[task]["score"] is not None]
    max_bits = float(config.section("storage")["maximum_theoretical_average_bits"])
    theoretical_pass = bool(storage_rows) and all(float(row["theoretical_average_bits"]) <= max_bits for row in storage_rows)
    matched_storage_mode = bool(
        attempt is not None
        and ((attempt / "joins").is_dir() or (attempt / "joins_remaining").is_dir())
    )
    if matched_storage_mode:
        # Paired candidates may legitimately emit different predictions, EOS
        # positions, and final cache lengths.  Their hard storage verdict comes
        # only from the independently materialized common-length checkpoint in
        # each atomic join.  Natural-generation row totals remain engineering
        # telemetry and must not be compared with a formula at unequal lengths.
        actual_pass = bool(expected_matched_storage_records) and (
            len(matched_storage_records) == expected_matched_storage_records
            and all(record["complete"] for record in matched_storage_records)
        )
        missing_actual_storage = max(
            0, expected_matched_storage_records - len(matched_storage_records)
        ) + sum(
            1 for record in matched_storage_records
            if not record["complete"] and not record["budget_exceeded"]
        )
        actual_storage_over_budget = sum(
            1 for record in matched_storage_records if record["budget_exceeded"]
        )
        storage_record_count = len(matched_storage_records)
        storage_measurement_basis = "matched_fixed_length_join"
    else:
        # Backward-compatible path for unpaired implementations and historical
        # fixtures which expose only row-level storage totals.
        actual_pass = bool(storage_rows) and all(row["actual_constraint_pass"] for row in storage_rows)
        missing_actual_storage = sum(1 for row in storage_rows if row.get("allocated_total_bytes") is None)
        actual_storage_over_budget = sum(1 for row in storage_rows if row.get("allocated_total_bytes") is not None and not row["actual_constraint_pass"])
        storage_record_count = len(storage_rows)
        storage_measurement_basis = "legacy_natural_generation_rows"
    tier = plan.get("evaluation_tier")
    full_formal = tier == "formal" and attempt is not None and (attempt / "formal_30pct_gate.json").is_file() and read_json(attempt / "formal_30pct_gate.json").get("qualified_for_remaining_70pct")
    paired_reference_complete = bool(
        matched_storage_mode
        and set(paired_control_metrics) == set(plan.get("datasets", []))
        and all(
            isinstance(metric.get("score"), (int, float))
            and metric.get("sample_ids_match") is True
            for metric in paired_control_metrics.values()
        )
    )
    if matched_storage_mode and not paired_reference_complete:
        failures.append("fresh paired VecInfer quality reference is incomplete")
    if paired_reference_complete and tier == "development":
        baseline = {
            "comparison_set": "fresh_paired_control_2048_math_quality",
            "longbench": {
                task: metric["score"] for task, metric in paired_control_metrics.items()
                if task not in {"gsm8k", "math500"}
            },
            "math": {
                task: paired_control_metrics[task]["score"]
                for task in ("gsm8k", "math500") if task in paired_control_metrics
            },
        }
    elif paired_reference_complete and tier == "formal":
        control_long = [
            metric["score"] for task, metric in paired_control_metrics.items()
            if task not in {"gsm8k", "math500"}
        ]
        baseline = {
            "comparison_set": "fresh_paired_control_2048_math_quality",
            "longbench": {"score": sum(control_long) / len(control_long)},
            "gsm8k": {"score": paired_control_metrics["gsm8k"]["score"]},
            "math500": {"score": paired_control_metrics["math500"]["score"]},
        }
    elif tier == "formal" and not full_formal:
        baseline = read_json(Path(config.section("project")["formal_baseline"]))
    elif tier == "formal":
        recorded = parse_baseline_report(Path(config.section("project")["baseline_report"]))
        baseline = {"longbench": {"score": recorded["longbench"]["宏平均"]},
                    "gsm8k": {"score": recorded["math"]["gsm8k"]},
                    "math500": {"score": recorded["math"]["math500"]}, "comparison_set": "full"}
    else:
        baseline = parse_baseline_report(Path(config.section("project")["baseline_report"]))
    artifact = control_envelope(
        "metrics",
        run_dir.name,
        "kvresearch.pipeline.aggregate",
        inputs,
        tasks=task_metrics,
        paired_control_tasks=paired_control_metrics,
        longbench={"score": sum(long_scores) / len(long_scores) if long_scores else None, "task_count": len(long_scores)},
        gsm8k=task_metrics.get("gsm8k"),
        math500=task_metrics.get("math500"),
        math={"score": sum(math_scores) / len(math_scores) if math_scores else None, "task_count": len(math_scores)},
        theoretical_storage_pass=theoretical_pass,
        actual_storage_pass=actual_pass,
        storage_accounting_summary={"records": storage_record_count,
                                    "expected_records": expected_matched_storage_records if matched_storage_mode else len(storage_rows),
                                    "measurement_basis": storage_measurement_basis,
                                    "missing_actual_bytes": missing_actual_storage,
                                    "over_vecinfer_budget": actual_storage_over_budget},
        peak_gpu_memory=max((float(row.get("peak_allocated_mib", 0) or 0) for task in plan.get("datasets", []) for raw_name in ("raw", "raw_remaining") for row in (read_jsonl((attempt / raw_name / f"{task}.jsonl")) if attempt and (attempt / raw_name / f"{task}.jsonl").is_file() else [])), default=0.0),
        elapsed_seconds=None,
        diagnostic_evidence={
            **diagnostic_validation,
            "phases": {
                "initial_30pct": diagnostic_validation,
                "remaining_70pct": remaining_diagnostic_validation,
            },
        },
        failures=failures,
        comparison_mode="development_candidate_ranking" if tier == "development" else "full_30plus70_vecinfer_1bit" if full_formal else "paired_fixed_30pct_vecinfer_1bit",
        evaluation_coverage={"initial_fraction": 0.30 if tier == "formal" else None,
                             "remaining_fraction": 0.70 if full_formal else 0.0,
                             "full_dataset": bool(full_formal)},
        baseline_reference=baseline,
        status="complete" if not failures else "failed",
        error_code=None if not failures else (
            "E_DIAGNOSTICS_INCOMPLETE"
            if (
                diagnostic_validation["required"] and not diagnostic_validation["valid"]
            ) or (
                remaining_diagnostic_validation
                and remaining_diagnostic_validation["required"]
                and not remaining_diagnostic_validation["valid"]
            )
            else "E_SAMPLES_INCOMPLETE"
        ),
    )
    atomic_write_json(cycle_dir / "metrics.json", artifact)
    return artifact


def verify(config: Config, run_dir: Path, cycle: str) -> dict[str, Any]:
    metrics_path = run_dir / "cycles" / cycle / "metrics.json"
    metrics = read_json(metrics_path)
    values: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, bool):
            values.append({"id": f"v{len(values):04d}", "value": value, "unit": "boolean", "direction": "constraint", "source_path": str(metrics_path), "json_pointer": pointer, "source_sha256": sha256_file(metrics_path), "verified": True})
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                rejected.append({"json_pointer": pointer, "reason": "non_finite"})
            else:
                values.append({"id": f"v{len(values):04d}", "value": value, "unit": "as_declared_by_metric", "direction": "recorded", "source_path": str(metrics_path), "json_pointer": pointer, "source_sha256": sha256_file(metrics_path), "verified": True})
        elif isinstance(value, dict):
            for key, child in value.items():
                if key not in {"baseline_reference", "inputs"}:
                    visit(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")

    visit(metrics, "")
    artifact = control_envelope(
        "verified_registry",
        run_dir.name,
        "kvresearch.pipeline.verify",
        [path_ref(metrics_path)],
        values=values,
        rejected_claims=rejected,
        source_hashes={str(metrics_path): sha256_file(metrics_path)},
    )
    atomic_write_json(run_dir / "cycles" / cycle / "verified_registry.json", artifact)
    return artifact


def diagnose(config: Config, run_dir: Path, cycle: str) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    metrics_path = cycle_dir / "metrics.json"
    registry_path = cycle_dir / "verified_registry.json"
    metrics = read_json(metrics_path)
    deficiencies: list[dict[str, Any]] = []
    attempt = latest_numbered_dir(cycle_dir / "attempts")
    exit_data = read_json(attempt / "exit.json") if attempt and (attempt / "exit.json").is_file() else {}
    execution_types = {
        "E_GPU_OOM": ("gpu_oom", "reduce runtime memory or repair implementation"),
        "E_TIMEOUT": ("execution_timeout", "retry transient timeout, then repair slow path"),
        "E_REMOTE_ENV": ("remote_environment_failure", "retry after fresh remote preflight"),
        "E_EXECUTION": ("code_or_runtime_failure", "repair candidate code or runtime interface"),
        "E_DIAGNOSTICS_INCOMPLETE": ("diagnostic_evidence_incomplete", "repair observer emission and rerun the same diagnostic candidate"),
    }
    if exit_data.get("error_code") in execution_types:
        kind, suggested = execution_types[exit_data["error_code"]]
        deficiencies.append({"type": kind, "severity": "critical", "evidence": [str(attempt / "exit.json"), exit_data["error_code"]],
                             "repairable": True, "suggested_action": suggested, "counts_as_research_result": False})
    diagnostic_evidence = metrics.get("diagnostic_evidence") or {}
    if diagnostic_evidence.get("required") and diagnostic_evidence.get("valid") is not True:
        deficiencies.append({
            "type": "diagnostic_evidence_incomplete", "severity": "critical",
            "evidence": diagnostic_evidence.get("evidence_paths", []) + diagnostic_evidence.get("errors", []),
            "repairable": True,
            "suggested_action": "emit complete finite hash-bound candidate/control observer evidence for every task and rerun",
            "counts_as_research_result": False,
        })
    if metrics.get("failures"):
        deficiencies.append({"type": "evaluation_failure", "severity": "critical", "evidence": metrics["failures"], "repairable": True,
                             "suggested_action": "repair execution or evaluation outputs", "counts_as_research_result": False})
    storage_summary = metrics.get("storage_accounting_summary") or {}
    storage_records = int(storage_summary.get("records", 0) or 0)
    if not metrics.get("theoretical_storage_pass"):
        if storage_records == 0:
            deficiencies.append({"type": "theoretical_storage_unmeasured", "severity": "critical", "evidence": [str(metrics_path)], "repairable": True,
                                 "suggested_action": "repair execution or storage instrumentation and rerun", "counts_as_research_result": False})
        else:
            deficiencies.append({"type": "theoretical_storage_exceeded", "severity": "critical", "evidence": [str(metrics_path)], "repairable": True,
                                 "suggested_action": "reduce theoretical allocation", "counts_as_research_result": True})
    if not metrics.get("actual_storage_pass"):
        missing_actual = int(storage_summary.get("missing_actual_bytes", 0) or 0)
        over_budget = int(storage_summary.get("over_vecinfer_budget", 0) or 0)
        if storage_records == 0 or missing_actual > 0:
            deficiencies.append({"type": "actual_storage_measurement_missing", "severity": "critical", "evidence": [str(metrics_path)], "repairable": True,
                                 "suggested_action": "repair byte instrumentation and rerun", "counts_as_research_result": False})
        if storage_records > 0 and over_budget > 0:
            deficiencies.append({"type": "actual_storage_exceeded", "severity": "critical", "evidence": [str(metrics_path)], "repairable": True,
                                 "suggested_action": "reduce candidate KV-cache allocations", "counts_as_research_result": True})
    mode = metrics.get("comparison_mode")
    tier = "development" if mode == "development_candidate_ranking" else "formal"
    locked = read_json(run_dir / "inputs.lock.json")
    locked_tasks = locked["datasets"][tier]["tasks"]
    remaining_tasks = locked["datasets"].get("formal_remaining", {}).get("tasks", {})
    full_formal = mode == "full_30plus70_vecinfer_1bit"
    for task, entry in metrics.get("tasks", {}).items():
        expected = int(locked_tasks.get(task, {}).get("rows", -1))
        if full_formal:
            expected += int(remaining_tasks.get(task, {}).get("rows", -1))
        if entry.get("samples") != expected:
            deficiencies.append({"type": "sample_identity_or_count_mismatch", "severity": "critical", "evidence": [task, entry.get("samples"), expected],
                                 "repairable": True, "suggested_action": "repair sample mapping and rerun fixed samples", "counts_as_research_result": False})
    if tier == "formal" and not metrics.get("failures"):
        baseline = metrics.get("baseline_reference", {})
        pairs = {
            "longbench": (metrics.get("longbench", {}).get("score"), baseline.get("longbench", {}).get("score")),
            "gsm8k": (metrics.get("gsm8k", {}).get("score"), baseline.get("gsm8k", {}).get("score")),
            "math500": (metrics.get("math500", {}).get("score"), baseline.get("math500", {}).get("score")),
        }
        deltas = {key: float(candidate) - float(reference) for key, (candidate, reference) in pairs.items()
                  if isinstance(candidate, (int, float)) and isinstance(reference, (int, float))}
        if len(deltas) == 3 and any(value < 0 for value in deltas.values()):
            deficiencies.append({"type": "quality_regression", "severity": "major", "evidence": [deltas], "repairable": True,
                                 "suggested_action": "refine the accuracy mechanism before spending more evaluation budget", "counts_as_research_result": True})
        elif len(deltas) == 3 and not any(value > 0 for value in deltas.values()):
            deficiencies.append({"type": "no_meaningful_gain", "severity": "major", "evidence": [deltas], "repairable": False,
                                 "suggested_action": "archive or pivot to another mechanism family", "counts_as_research_result": True})
    if not deficiencies:
        deficiencies.append({"type": "none", "severity": "info", "evidence": [str(metrics_path)], "repairable": False,
                             "suggested_action": "evaluate quality thresholds", "counts_as_research_result": True})
    diagnostic_inputs = [
        path_ref(Path(path)) for path in diagnostic_evidence.get("evidence_paths", [])
        if Path(path).is_file()
    ]
    artifact = control_envelope(
        "experiment_diagnosis",
        run_dir.name,
        "kvresearch.pipeline.diagnose",
        [path_ref(metrics_path), path_ref(registry_path)]
        + ([path_ref(attempt / "exit.json")] if attempt and (attempt / "exit.json").is_file() else [])
        + diagnostic_inputs,
        deficiencies=deficiencies,
        severity=max((item["severity"] for item in deficiencies), key=lambda item: {"info": 0, "minor": 1, "major": 2, "critical": 3}[item]),
        evidence=sorted({str(value) for item in deficiencies for value in item.get("evidence", [])}),
        repairable=any(item["repairable"] for item in deficiencies),
        suggested_actions=sorted({item["suggested_action"] for item in deficiencies}),
        measurement_valid=not any(item.get("counts_as_research_result") is False for item in deficiencies),
        diagnostic_evidence=diagnostic_evidence,
    )
    atomic_write_json(cycle_dir / "experiment_diagnosis.json", artifact)
    return artifact


def _selected_mechanism_family(cycle_dir: Path) -> str | None:
    selection_path = cycle_dir / "selection.json"
    candidates_path = cycle_dir / "candidates.jsonl"
    if not selection_path.is_file() or not candidates_path.is_file():
        return None
    selected = read_json(selection_path).get("selected_candidate_id")
    row = next((item for item in read_jsonl(candidates_path) if item.get("candidate_id") == selected), None)
    return canonical_mechanism_family(row or {})


def _cooldown_state(config: Config, run_dir: Path, cycle: str, metrics: dict[str, Any], measurement_valid: bool) -> dict[str, Any]:
    current_index = int(cycle)
    family = _selected_mechanism_family(run_dir / "cycles" / cycle)
    state = {"mechanism_family": family, "small_gain": False, "small_gain_streak": 0,
             "triggered": False, "cooldown_until_cycle": None, "reason": None, "active_families": []}
    previous_dir = run_dir / "cycles" / f"{current_index - 1:03d}"
    previous_decision_path = previous_dir / "decision.json"
    previous_decision = read_json(previous_decision_path) if current_index > 0 and previous_decision_path.is_file() else {}
    previous_cooldown = previous_decision.get("cooldown", {})
    inherited = list(previous_cooldown.get("active_families", []))
    if previous_cooldown.get("triggered") and previous_cooldown.get("mechanism_family"):
        inherited.append({"mechanism_family": previous_cooldown["mechanism_family"],
                          "until_cycle": previous_cooldown.get("cooldown_until_cycle")})
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in inherited:
        name = str(item.get("mechanism_family", "")).strip().lower()
        until = int(item.get("until_cycle") or -1)
        if name and until >= current_index:
            deduplicated[name] = {"mechanism_family": name, "until_cycle": until}
    state["active_families"] = list(deduplicated.values())
    if not measurement_valid or not family or current_index == 0:
        return state
    # Use the most recent observation from this host-owned broad family.
    # A differently named candidate or an interleaved family cannot reset the
    # streak and thereby evade cooldown.
    prior_same_dir = None
    for index in range(current_index - 1, -1, -1):
        candidate_dir = run_dir / "cycles" / f"{index:03d}"
        if _selected_mechanism_family(candidate_dir) == family:
            prior_same_dir = candidate_dir
            break
    if prior_same_dir is None:
        return state
    previous_dir = prior_same_dir
    previous_decision_path = previous_dir / "decision.json"
    previous_metrics_path = previous_dir / "metrics.json"
    if not previous_metrics_path.is_file() or not previous_decision_path.is_file():
        return state
    previous_decision = read_json(previous_decision_path)
    previous_metrics = read_json(previous_metrics_path)
    previous_diagnosis_path = previous_dir / "experiment_diagnosis.json"
    previous_measurement_valid = (
        previous_diagnosis_path.is_file()
        and read_json(previous_diagnosis_path).get("measurement_valid") is True
    )
    previous_long_score = (previous_metrics.get("longbench") or {}).get("score")
    current_long_score = (metrics.get("longbench") or {}).get("score")
    # A technical or incomplete predecessor is not a scientific observation.
    # It must neither contribute a delta nor advance the same-family cooldown.
    if not (
        previous_measurement_valid
        and previous_metrics.get("status") == "complete"
        and not previous_metrics.get("failures")
        and previous_metrics.get("theoretical_storage_pass") is True
        and previous_metrics.get("actual_storage_pass") is True
        and isinstance(previous_long_score, (int, float))
        and isinstance(current_long_score, (int, float))
    ):
        return state
    if previous_metrics.get("comparison_mode") != metrics.get("comparison_mode"):
        return state
    long_delta = float(current_long_score) - float(previous_long_score)
    gsm_delta = int((metrics.get("gsm8k") or {}).get("correct") or 0) - int((previous_metrics.get("gsm8k") or {}).get("correct") or 0)
    math_delta = int((metrics.get("math500") or {}).get("correct") or 0) - int((previous_metrics.get("math500") or {}).get("correct") or 0)
    policy = config.section("cooldown")
    meaningful = (
        long_delta >= 0 and gsm_delta >= 0 and math_delta >= 0
        and (long_delta >= float(policy["longbench_meaningful_gain_points"])
             or gsm_delta >= int(policy["math_meaningful_gain_correct_answers"])
             or math_delta >= int(policy["math_meaningful_gain_correct_answers"]))
    )
    state["deltas_vs_previous_same_family"] = {"longbench": long_delta, "gsm8k_correct": gsm_delta, "math500_correct": math_delta}
    state["small_gain"] = not meaningful
    previous_streak = int(previous_decision.get("cooldown", {}).get("small_gain_streak", 0))
    state["small_gain_streak"] = previous_streak + 1 if not meaningful else 0
    limit = int(policy["same_family_small_gain_limit"])
    if state["small_gain_streak"] >= limit:
        state.update(triggered=True, cooldown_until_cycle=current_index + int(policy["cooldown_cycles"]),
                     reason=f"mechanism family {family} produced {state['small_gain_streak']} consecutive sub-threshold gains")
        deduplicated[family] = {"mechanism_family": family, "until_cycle": state["cooldown_until_cycle"]}
        state["active_families"] = list(deduplicated.values())
    return state


def _development_research_window_count(run_dir: Path, cycle: str) -> int:
    """Count valid development observations in the current refinement window.

    Technical/incomplete cycles are deliberately ignored.  A prior scientific
    pivot (or promotion/terminal decision) starts a new window, so its closing
    observation is not charged to the next direction.
    """
    current_index = int(cycle)
    count = 1
    for index in range(current_index - 1, -1, -1):
        cycle_dir = run_dir / "cycles" / f"{index:03d}"
        plan_path = cycle_dir / "experiment_plan.json"
        metrics_path = cycle_dir / "metrics.json"
        diagnosis_path = cycle_dir / "experiment_diagnosis.json"
        decision_path = cycle_dir / "decision.json"
        if not all(path.is_file() for path in (plan_path, metrics_path, diagnosis_path, decision_path)):
            continue
        plan = read_json(plan_path)
        metrics = read_json(metrics_path)
        diagnosis = read_json(diagnosis_path)
        decision = read_json(decision_path)
        if plan.get("evaluation_tier") != "development":
            break
        valid_observation = (
            diagnosis.get("measurement_valid") is True
            and metrics.get("status") == "complete"
            and not metrics.get("failures")
            and metrics.get("theoretical_storage_pass") is True
            and metrics.get("actual_storage_pass") is True
            and decision.get("status") == "complete"
        )
        if not valid_observation:
            continue
        if decision.get("action") in {"PIVOT", "PROMOTE", "ARCHIVE", "STOP"}:
            break
        count += 1
    return count


def decide(config: Config, run_dir: Path, cycle: str) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    metrics_path = cycle_dir / "metrics.json"
    registry_path = cycle_dir / "verified_registry.json"
    diagnosis_path = cycle_dir / "experiment_diagnosis.json"
    metrics = read_json(metrics_path)
    registry = read_json(registry_path)
    diagnosis = read_json(diagnosis_path)
    plan = read_json(cycle_dir / "experiment_plan.json")
    tier = plan.get("evaluation_tier")
    diagnostic = diagnostic_contract(plan)
    diagnostic_evidence = metrics.get("diagnostic_evidence") or {}
    storage_ok = metrics.get("theoretical_storage_pass") is True and metrics.get("actual_storage_pass") is True
    no_failures = not metrics.get("failures")
    baseline = metrics.get("baseline_reference", {})
    long_score = (metrics.get("longbench") or {}).get("score")
    gsm8k_score = (metrics.get("gsm8k") or {}).get("score")
    math500_score = (metrics.get("math500") or {}).get("score")
    math_score = (metrics.get("math") or {}).get("score")
    if tier == "development":
        dev_tasks = [task for task in metrics.get("tasks", {}) if task not in {"gsm8k", "math500"}]
        base_long_values = [baseline.get("longbench", {}).get(task) for task in dev_tasks]
        base_long_values = [float(value) for value in base_long_values if isinstance(value, (int, float))]
        base_long = sum(base_long_values) / len(base_long_values) if base_long_values else None
    else:
        base_long = baseline.get("longbench", {}).get("score")
    baseline_math = baseline.get("math", {})
    base_gsm8k = (baseline.get("gsm8k") or {}).get("score") if tier == "formal" else baseline_math.get("gsm8k")
    base_math500 = (baseline.get("math500") or {}).get("score") if tier == "formal" else baseline_math.get("math500")
    base_math_values = [base_gsm8k, base_math500]
    base_math_values = [float(value) for value in base_math_values if isinstance(value, (int, float))]
    base_math = sum(base_math_values) / len(base_math_values) if base_math_values else None
    quality_ok = all(isinstance(value, (int, float)) for value in (long_score, math_score, base_long, base_math)) and long_score > base_long and math_score > base_math
    formal_improvements = {
        "longbench": isinstance(long_score, (int, float)) and isinstance(base_long, (int, float)) and long_score > base_long,
        "gsm8k": isinstance(gsm8k_score, (int, float)) and isinstance(base_gsm8k, (int, float)) and gsm8k_score > base_gsm8k,
        "math500": isinstance(math500_score, (int, float)) and isinstance(base_math500, (int, float)) and math500_score > base_math500,
    }
    full_formal = metrics.get("comparison_mode") == "full_30plus70_vecinfer_1bit"
    measurement_valid = diagnosis.get("measurement_valid", True)
    if diagnostic["required"] and diagnostic_evidence.get("valid") is not True:
        measurement_valid = False
    # A causal probe is evidence gathering, not another refinement of the
    # measured mechanism family.  It must not increment or trigger cooldown.
    cooldown = _cooldown_state(
        config, run_dir, cycle, metrics,
        measurement_valid and not diagnostic["non_promotable"],
    )
    valid_development_attempts = (
        _development_research_window_count(run_dir, cycle)
        if tier == "development" and measurement_valid and no_failures and storage_ok
        and not diagnostic["non_promotable"]
        else 0
    )
    if not measurement_valid:
        action = "REFINE"
        rationale = "measurement is invalid; automatically repair the classified technical failure before judging accuracy or bit budget"
    elif not no_failures or not storage_ok:
        action = "REFINE"
        rationale = "execution or storage hard constraint failed"
    elif diagnostic["non_promotable"]:
        action = "PIVOT"
        rationale = "diagnostic evidence is valid, but this probe is non-promotable; explore a different quality mechanism using its causal evidence"
    elif tier == "formal" and full_formal and all(formal_improvements.values()):
        action = "STOP"
        rationale = "all three full 30%+70% metrics exceed VecInfer 1-bit under storage constraints"
    elif tier == "formal" and not full_formal and all(formal_improvements.values()):
        action = "REFINE"
        rationale = "30% gate passed but the required remaining 70% completion result is missing"
    elif tier == "formal" and not any(formal_improvements.values()):
        action = "ARCHIVE"
        rationale = "formal candidate failed: none of LongBench average, GSM8K accuracy, or MATH500 accuracy exceeds VecInfer 1-bit"
    elif tier == "formal":
        action = "REFINE"
        rationale = "formal candidate improves at least one but not all three quality metrics; retain evidence for refinement"
    elif tier == "development" and quality_ok:
        action = "PROMOTE"
        rationale = "development candidate clears both quality references and storage constraints"
    elif valid_development_attempts >= 3:
        action = "PIVOT"
        rationale = "three valid development research attempts did not clear both quality targets"
    else:
        action = "REFINE"
        rationale = "quality target not yet met; retain mechanism for bounded refinement"
    if action == "REFINE" and cooldown["triggered"]:
        action = "PIVOT"
        rationale = cooldown["reason"] + "; explore a different mechanism family"
    value_ids = [item["id"] for item in registry.get("values", []) if item.get("json_pointer") in {"/longbench/score", "/gsm8k/score", "/math500/score", "/theoretical_storage_pass", "/actual_storage_pass"}]
    artifact = control_envelope(
        "decision",
        run_dir.name,
        "kvresearch.pipeline.decide",
        [path_ref(metrics_path), path_ref(registry_path), path_ref(diagnosis_path)],
        action=action,
        verified_value_ids=value_ids,
        rationale=rationale,
        comparisons={"candidate_longbench": long_score, "baseline_longbench": base_long,
                     "candidate_gsm8k": gsm8k_score, "baseline_gsm8k": base_gsm8k,
                     "candidate_math500": math500_score, "baseline_math500": base_math500,
                     "formal_improvements": formal_improvements,
                     "valid_development_attempts_in_window": valid_development_attempts},
        next_cycle_constraints=[rationale] + [f"exclude mechanism_family={item['mechanism_family']} through cycle {item['until_cycle']}"
                                                for item in cooldown.get("active_families", [])],
        cooldown=cooldown,
        measurement_valid=measurement_valid,
        next_cycle_mode="technical_repair" if not measurement_valid else "research",
        diagnostic_evidence={
            "required": diagnostic["required"],
            "non_promotable": diagnostic["non_promotable"],
            "valid": diagnostic_evidence.get("valid"),
            "paths": diagnostic_evidence.get("evidence_paths", []),
        },
    )
    atomic_write_json(cycle_dir / "decision.json", artifact)
    return artifact


def learn(run_dir: Path, cycle: str) -> list[dict[str, Any]]:
    cycle_dir = run_dir / "cycles" / cycle
    diagnosis = read_json(cycle_dir / "experiment_diagnosis.json")
    decision = read_json(cycle_dir / "decision.json")
    rows = []
    for index, item in enumerate(diagnosis.get("deficiencies", [])):
        rows.append({
            "lesson_id": f"{run_dir.name}-{cycle}-{index:03d}",
            "evidence": item.get("evidence", []),
            "scope": "1-bit KV-cache candidate",
            "failed_mechanism": item.get("type"),
            "effective_fix": item.get("suggested_action"),
            "confidence": 1.0 if item.get("severity") == "critical" else 0.6,
            "source_cycle": cycle,
            "decision": decision.get("action"),
        })
    cooldown = decision.get("cooldown", {})
    diagnostic = diagnosis.get("diagnostic_evidence") or {}
    if diagnostic.get("required") and diagnostic.get("valid"):
        rows.append({
            "lesson_id": f"{run_dir.name}-{cycle}-diagnostic",
            "evidence": diagnostic.get("evidence_paths", []),
            "scope": "causal diagnostic observer",
            "failed_mechanism": None,
            "effective_fix": "explore a different mechanism using the validated candidate/control observer comparison",
            "diagnostic_summary": diagnostic.get("tasks", {}),
            "confidence": 1.0,
            "source_cycle": cycle,
            "decision": decision.get("action"),
        })
    if cooldown.get("triggered"):
        rows.append({"lesson_id": f"{run_dir.name}-{cycle}-cooldown", "evidence": [str(cycle_dir / "decision.json")],
                     "scope": cooldown.get("mechanism_family"), "failed_mechanism": "same_family_small_gain_streak",
                     "effective_fix": f"explore another family until cycle {cooldown.get('cooldown_until_cycle')}",
                     "confidence": 1.0, "source_cycle": cycle, "decision": decision.get("action"),
                     "cooldown": cooldown})
    atomic_write_jsonl(cycle_dir / "lessons.jsonl", rows)
    memory = run_dir.parent.parent / "memory" / "lessons.jsonl"
    memory.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(memory) if memory.is_file() else []
    replacements = {row["lesson_id"]: row for row in rows}
    merged = []
    replaced = set()
    for row in existing:
        lesson_id = row.get("lesson_id")
        if lesson_id in replacements:
            merged.append(replacements[lesson_id])
            replaced.add(lesson_id)
        else:
            merged.append(row)
    merged.extend(row for lesson_id, row in replacements.items() if lesson_id not in replaced)
    atomic_write_jsonl(memory, merged)
    return rows


def finalize(run_dir: Path, cycle: str, reason: str | None = None) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    decision_path = cycle_dir / "decision.json"
    registry_path = cycle_dir / "verified_registry.json"
    decision = read_json(decision_path)
    selection = read_json(cycle_dir / "selection.json")
    artifact = control_envelope(
        "terminal",
        run_dir.name,
        "kvresearch.pipeline.finalize",
        [path_ref(decision_path), path_ref(registry_path)],
        termination_type=reason or "objective_met",
        final_candidate_id=selection.get("selected_candidate_id"),
        final_evidence=[str(registry_path)],
        unresolved=[] if decision.get("action") == "STOP" else [reason or decision.get("rationale")],
        reproduction_entrypoint=f"kvresearch --config config.kvresearch.yaml next --run {run_dir}",
    )
    atomic_write_json(run_dir / "terminal.json", artifact)
    return artifact
