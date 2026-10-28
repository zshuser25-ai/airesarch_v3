from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from kvresearch.config import Config
from kvresearch.diagnostics import validate_diagnostic_outputs
from kvresearch.stock_control import validate_attempt_stock_controls
from kvresearch.stock_control import validate_stock_control_file
from kvresearch.remote import RemoteCleanupError, RemotePod
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref, read_json, read_jsonl, sha256_file, utcnow
from kvresearch.vecinfer import VecInferAdapter


_COMMIT_VISIBILITY_TIMEOUT_SECONDS = 90.0
_COMMIT_VISIBILITY_POLL_SECONDS = 1.0


def _with_batch_size(spec, batch_size: int):
    argv = list(spec.argv)
    index = argv.index("--batch_size") + 1
    argv[index] = str(batch_size)
    return replace(spec, argv=argv)


def _effective_batch_attempts(batch_sizes: list[int], concurrency_cap: int) -> list[int]:
    """Keep one requested batch for each distinct effective concurrency."""
    if concurrency_cap < 1:
        raise ValueError("request concurrency cap must be positive")
    result = []
    seen = set()
    for requested in batch_sizes:
        effective = min(int(requested), concurrency_cap)
        if effective in seen:
            continue
        seen.add(effective)
        result.append(effective)
    return result


def _wait_for_committed_file(path: Path, timeout_seconds: float,
                             poll_seconds: float = _COMMIT_VISIBILITY_POLL_SECONDS) -> str | None:
    """Wait for an atomically published artifact to become visible on Host NFS.

    The remote worker and Host use different NFS clients.  A prior negative
    directory lookup on Host can remain cached briefly after the remote worker
    has completed an atomic rename, so an immediate ``is_file`` check can
    falsely reject a successful shard.
    """
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            if path.is_file():
                return sha256_file(path)
        except OSError:
            # Publication is atomic, but tolerate a transient NFS lookup/read
            # error until the bounded visibility window expires.
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(max(0.0, poll_seconds), remaining))


def _bind_gpu(spec, gpu: dict[str, Any]):
    env = dict(spec.env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    return replace(spec, env=env, gpu_index=int(gpu["index"]), gpu_uuid=str(gpu.get("uuid", "")))


def _replace_option(argv: list[str], name: str, value: str | Path) -> list[str]:
    result = list(argv)
    position = result.index(name) + 1
    result[position] = str(value)
    return result


def _locked_row_counts(specs: list) -> dict[str, int]:
    """Read row counts from the hash-locked staged-dataset manifest."""
    if not specs:
        return {}
    roots = {Path(spec.env["KVRESEARCH_DATASETS_ROOT"]).resolve() for spec in specs}
    if len(roots) != 1:
        raise RuntimeError("sample sharding requires one locked dataset root")
    marker = next(iter(roots)) / ".complete.json"
    value = read_json(marker)
    if value.get("status") != "complete":
        raise RuntimeError(f"staged dataset marker is incomplete: {marker}")
    tasks = (value.get("manifest") or {}).get("tasks") or {}
    counts: dict[str, int] = {}
    for spec in specs:
        task = spec.name.removeprefix("generate-")
        rows = ((tasks.get(task) or {}).get("output") or {}).get("rows")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise RuntimeError(f"locked row count is missing for {task}: {marker}")
        counts[task] = rows
    return counts


def _sample_shard_specs(specs: list, attempt_dir: Path, phase: str,
                        chunk_size: int,
                        task_chunk_sizes: dict[str, int] | None = None) -> tuple[list, dict[str, list]]:
    """Expand task commands into deterministic, disjoint sample-range units."""
    if chunk_size < 1:
        raise ValueError("formal sample chunk size must be positive")
    counts = _locked_row_counts(specs)
    units = []
    grouped: dict[str, list] = {}
    namespace = "raw_remaining" if phase == "remaining_70pct" else "raw"
    for original in specs:
        task = original.name.removeprefix("generate-")
        task_chunk_size = int((task_chunk_sizes or {}).get(task, chunk_size))
        if task_chunk_size < 1:
            raise ValueError(f"formal sample chunk size must be positive for {task}")
        final_output = original.raw_output
        if final_output is None:
            raise RuntimeError(f"sample-sharded command lacks raw output: {task}")
        for start in range(0, counts[task], task_chunk_size):
            end = min(counts[task], start + task_chunk_size)
            unit_id = f"{task}:{start}-{end}"
            shard_root = (
                attempt_dir / "sample_shards" / phase / task
                / f"{start:06d}-{end:06d}"
            )
            output = shard_root / namespace / f"{task}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            env = dict(original.env)
            env.update({
                "VECINFER_QUESTION_START": str(start),
                "VECINFER_MAX_QUESTIONS": str(end - start),
                "KVRESEARCH_SAMPLE_SHARD": "1",
                "KVRESEARCH_SAMPLE_SHARD_ID": unit_id,
                "KVRESEARCH_SAMPLE_SHARD_PROBE_OWNER": "1" if start == 0 else "0",
            })
            unit = replace(
                original,
                name=f"generate-{task}--samples-{start}-{end}",
                argv=_replace_option(original.argv, "--save_path", output),
                env=env,
                raw_output=output,
                task_name=task,
                unit_id=unit_id,
                sample_start=start,
                sample_end=end,
                final_output=final_output,
            )
            units.append(unit)
            grouped.setdefault(task, []).append(unit)
    return units, grouped


def _sample_shard_registry(specs: list, phase: str, chunk_size: int,
                           task_chunk_sizes: dict[str, int] | None = None) -> list[dict[str, Any]]:
    counts = _locked_row_counts(specs)
    registry = []
    for spec in specs:
        task = spec.name.removeprefix("generate-")
        task_chunk_size = int((task_chunk_sizes or {}).get(task, chunk_size))
        if task_chunk_size < 1:
            raise ValueError(f"formal sample chunk size must be positive for {task}")
        for start in range(0, counts[task], task_chunk_size):
            end = min(counts[task], start + task_chunk_size)
            registry.append({
                "unit_id": f"{task}:{start}-{end}", "phase": phase,
                "task": task, "sample_start": start, "sample_end": end,
                "sample_count": end - start, "probe_owner": start == 0,
                "assignment": "dynamic_at_runtime",
            })
    return registry


def _run_task(adapter: VecInferAdapter, original, attempt_dir: Path, batch_sizes: list[int],
              phase: str, worker_slot: int) -> dict[str, Any]:
    failed_raw = attempt_dir / "failed_raw"
    task = original.task_name or original.name.removeprefix("generate-")
    log_unit = (original.unit_id or task).replace(":", ".").replace("/", "_")
    stdout_path = attempt_dir / "logs" / f"{phase}.{log_unit}.stdout.log"
    stderr_path = attempt_dir / "logs" / f"{phase}.{log_unit}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = batch_sizes if batch_sizes else [int(original.argv[original.argv.index("--batch_size") + 1])]
    record: dict[str, Any] = {
        "task": task, "phase": phase, "gpu_index": original.gpu_index,
        "gpu_uuid": original.gpu_uuid, "worker_slot": worker_slot,
        "unit_id": original.unit_id or task,
        "sample_start": original.sample_start, "sample_end": original.sample_end,
        "returncode": 1, "timed_out": False, "oom": False, "batch_attempts": [],
        "started_at": utcnow(),
    }
    for batch_size in attempts:
        spec = _with_batch_size(original, batch_size)
        execution_env = dict(spec.env)
        execution_env["KVRESEARCH_REMOTE_REGISTRY_DIR"] = str(
            attempt_dir / "remote_processes"
        )
        execution_env["KVRESEARCH_REMOTE_EXECUTION_UNIT"] = str(
            original.unit_id or task
        )
        spec = replace(spec, env=execution_env)
        before = adapter.remote.runtime_sample(spec.gpu_uuid, f"before:{spec.name}:batch{batch_size}")
        started = time.monotonic()
        try:
            result = adapter.remote.execute_argv(spec.argv, spec.cwd, spec.env, spec.timeout_seconds)
            is_oom = "out of memory" in result.stderr.lower()
            command_timed_out = result.returncode == 124
            with stdout_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utcnow()}] batch_size={batch_size} returncode={result.returncode}\n{result.stdout}")
            with stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utcnow()}] batch_size={batch_size} returncode={result.returncode}\n{result.stderr}")
            record["batch_attempts"].append({"batch_size": batch_size, "returncode": result.returncode,
                                             "oom": is_oom, "timed_out": command_timed_out,
                                             "elapsed_seconds": time.monotonic() - started})
            record.update(returncode=result.returncode, oom=is_oom,
                          timed_out=command_timed_out, batch_size=batch_size)
            if result.returncode == 0:
                if spec.raw_output is not None and spec.sample_start is not None:
                    join_namespace = "joins_remaining" if spec.raw_output.parent.name == "raw_remaining" else "joins"
                    join_path = spec.raw_output.parent.parent / join_namespace / f"{task}.json"
                    commit_wait_started = time.monotonic()
                    commit_sha = _wait_for_committed_file(
                        join_path, _COMMIT_VISIBILITY_TIMEOUT_SECONDS
                    )
                    record["commit_visibility_wait_seconds"] = (
                        time.monotonic() - commit_wait_started
                    )
                    if commit_sha is None:
                        record.update(returncode=1, oom=False)
                        record["closure_error"] = (
                            "successful shard lacks atomic join after bounded NFS visibility wait: "
                            f"{join_path}"
                        )
                        break
                    record["commit"] = {
                        "path": str(join_path), "sha256": commit_sha,
                    }
                break
            if not is_oom:
                break
            if spec.raw_output and spec.raw_output.exists():
                failed_raw.mkdir(parents=True, exist_ok=True)
                failed_name = f"{log_unit}.batch{batch_size}.jsonl"
                shutil.move(str(spec.raw_output), str(failed_raw / failed_name))
        except subprocess.TimeoutExpired:
            record.update(returncode=124, timed_out=True)
            record["batch_attempts"].append({"batch_size": batch_size, "returncode": 124,
                                             "oom": False, "elapsed_seconds": time.monotonic() - started})
            break
        except RemoteCleanupError as exc:
            record.update(returncode=125, timed_out=True, cleanup_failed=True)
            record["closure_error"] = str(exc)
            record["batch_attempts"].append({
                "batch_size": batch_size, "returncode": 125, "oom": False,
                "timed_out": True, "cleanup_failed": True,
                "elapsed_seconds": time.monotonic() - started,
            })
            break
        finally:
            record.setdefault("runtime_trace", []).extend(
                [before, adapter.remote.runtime_sample(spec.gpu_uuid, f"after:{spec.name}:batch{batch_size}")]
            )
    record["ended_at"] = utcnow()
    return record


def _merge_sample_shards(adapter: VecInferAdapter, original, units: list,
                         attempt_dir: Path, phase: str) -> dict[str, Any]:
    """Ask the isolated implementation to attest and atomically merge a task."""
    task = original.name.removeprefix("generate-")
    namespace = "raw_remaining" if phase == "remaining_70pct" else "raw"
    manifest_path = attempt_dir / "sample_shards" / phase / task / "merge_manifest.json"
    dataset_marker = Path(original.env["KVRESEARCH_DATASETS_ROOT"]) / ".complete.json"
    manifest = {
        "format_version": 1,
        "task": task,
        "phase": phase,
        "expected_rows": max(int(unit.sample_end) for unit in units),
        "final_output": str(original.raw_output),
        "dataset_marker": path_ref(dataset_marker),
        "shards": [
            {
                "unit_id": unit.unit_id,
                "sample_start": unit.sample_start,
                "sample_end": unit.sample_end,
                "join": str(unit.raw_output.parent.parent / (
                    "joins_remaining" if namespace == "raw_remaining" else "joins"
                ) / f"{task}.json"),
            }
            for unit in units
        ],
    }
    atomic_write_json(manifest_path, manifest)
    env = dict(original.env)
    env["KVRESEARCH_SHARD_MERGE_MANIFEST"] = str(manifest_path)
    env["KVRESEARCH_REMOTE_REGISTRY_DIR"] = str(attempt_dir / "remote_processes")
    env["KVRESEARCH_REMOTE_EXECUTION_UNIT"] = f"{task}:merge"
    started = time.monotonic()
    timed_out = False
    try:
        result = adapter.remote.execute_argv(
            original.argv, original.cwd, env, original.timeout_seconds
        )
        timed_out = result.returncode == 124
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        result = subprocess.CompletedProcess(
            original.argv, 124,
            exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        )
    except RemoteCleanupError as exc:
        timed_out = True
        result = subprocess.CompletedProcess(original.argv, 125, "", str(exc))
    stdout_path = attempt_dir / "logs" / f"{phase}.{task}.merge.stdout.log"
    stderr_path = attempt_dir / "logs" / f"{phase}.{task}.merge.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "task": task, "phase": phase, "unit_id": f"{task}:merge",
        "returncode": result.returncode, "timed_out": timed_out,
        "cleanup_failed": result.returncode == 125,
        "oom": "out of memory" in result.stderr.lower(),
        "elapsed_seconds": time.monotonic() - started,
        "merge_manifest": str(manifest_path),
        "sample_shards": len(units),
    }


def _run_sample_sharded_gpu_pool(adapter: VecInferAdapter, specs: list,
                                  approved_gpus: list[dict[str, Any]],
                                  attempt_dir: Path, batch_sizes: list[int],
                                  phase: str, chunk_size: int,
                                  task_chunk_sizes: dict[str, int] | None = None) -> list[dict[str, Any]]:
    units, grouped = _sample_shard_specs(
        specs, attempt_dir, phase, chunk_size, task_chunk_sizes
    )
    unit_records = _run_dynamic_gpu_pool(
        adapter, units, approved_gpus, attempt_dir, batch_sizes, phase,
        journal_path=attempt_dir / "scheduler_journal.jsonl",
    )
    records_by_unit = {record["unit_id"]: record for record in unit_records}
    cancel_cleanup_failed = any(
        record.get("cleanup_failed", False) for record in unit_records
    )
    task_records: list[dict[str, Any]] = []
    merge_cancelled = False
    for original in specs:
        task = original.name.removeprefix("generate-")
        task_units = grouped[task]
        executions = [
            records_by_unit[unit.unit_id]
            for unit in task_units if unit.unit_id in records_by_unit
        ]
        if merge_cancelled or len(executions) != len(task_units):
            task_records.append({
                "task": task, "phase": phase, "unit_id": f"{task}:merge-skipped",
                "returncode": 125 if cancel_cleanup_failed else 130,
                "timed_out": False, "oom": False,
                "cleanup_failed": cancel_cleanup_failed,
                "shard_executions": executions,
                "closure_error": (
                    "dynamic queue cancelled after unconfirmed remote cleanup"
                    if cancel_cleanup_failed else "dynamic queue cancelled after peer failure"
                ),
                "runtime_trace": [
                    sample for item in executions
                    for sample in item.get("runtime_trace", [])
                ],
            })
            continue
        failed = next((record for record in executions if record["returncode"] != 0), None)
        if failed is not None:
            task_records.append({
                "task": task, "phase": phase, "unit_id": f"{task}:merge-skipped",
                "returncode": failed["returncode"], "timed_out": failed["timed_out"],
                "oom": failed["oom"], "shard_executions": executions,
                "runtime_trace": [sample for item in executions for sample in item.get("runtime_trace", [])],
            })
            continue
        merged = _merge_sample_shards(adapter, original, task_units, attempt_dir, phase)
        if merged["returncode"] != 0:
            merge_cancelled = True
            cleanup_confirmed = adapter.remote.cleanup_registry(
                attempt_dir / "remote_processes"
            )
            merged["peer_cleanup_confirmed"] = cleanup_confirmed
            if not cleanup_confirmed:
                merged.update(returncode=125, cleanup_failed=True)
        merged["shard_executions"] = [
            {key: value for key, value in item.items() if key != "runtime_trace"}
            for item in executions
        ]
        merged["runtime_trace"] = [
            sample for item in executions for sample in item.get("runtime_trace", [])
        ]
        merged["gpu_indices"] = sorted({int(item["gpu_index"]) for item in executions})
        merged["worker_slots"] = sorted({int(item["worker_slot"]) for item in executions})
        task_records.append(merged)
    return task_records


def _run_dynamic_gpu_pool(adapter: VecInferAdapter, specs: list, approved_gpus: list[dict[str, Any]],
                          attempt_dir: Path, batch_sizes: list[int], phase: str,
                          journal_path: Path | None = None) -> list[dict[str, Any]]:
    """Run a shared task queue with one model process per GPU.

    A worker immediately takes the next unclaimed task after finishing, so a
    short shard can no longer leave its GPU idle while another GPU owns a long
    static tail. Intra-task sample concurrency is controlled by ``batch_sizes``.
    """
    if not specs:
        return []
    if not approved_gpus:
        raise ValueError("dynamic scheduler requires at least one approved GPU")
    work: Queue = Queue()
    for spec in specs:
        work.put(spec)
    journal_lock = threading.Lock()
    cleanup_lock = threading.Lock()
    cancel_event = threading.Event()
    registry_dir = attempt_dir / "remote_processes"
    registry_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    registry_dir.chmod(0o755)
    journal_records = read_jsonl(journal_path) if journal_path and journal_path.is_file() else []

    def commit_journal(record: dict[str, Any]) -> None:
        if journal_path is None:
            return
        durable = {key: value for key, value in record.items() if key != "runtime_trace"}
        with journal_lock:
            journal_records.append(durable)
            atomic_write_jsonl(journal_path, journal_records)

    def worker(gpu: dict[str, Any], worker_slot: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        while True:
            if cancel_event.is_set():
                return records
            try:
                pending = work.get_nowait()
            except Empty:
                return records
            try:
                if cancel_event.is_set():
                    return records
                bound = _bind_gpu(pending, gpu)
                record = _run_task(adapter, bound, attempt_dir, batch_sizes, phase, worker_slot)
                records.append(record)
                if record["returncode"] != 0:
                    cancel_event.set()
                    with cleanup_lock:
                        cleanup_confirmed = adapter.remote.cleanup_registry(registry_dir)
                    record["peer_cleanup_confirmed"] = cleanup_confirmed
                    if not cleanup_confirmed:
                        record.update(returncode=125, cleanup_failed=True)
                        record["closure_error"] = (
                            "E_REMOTE_CLEANUP: peer process groups could not be proven terminated"
                        )
                commit_journal(record)
            finally:
                work.task_done()

    worker_count = min(len(approved_gpus), len(specs))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(worker, gpu, slot) for slot, gpu in enumerate(approved_gpus[:worker_count])]
        nested = [future.result() for future in futures]
    return [record for group in nested for record in group]


def _formal_30pct_gate(config: Config, attempt_dir: Path, tasks: list[str]) -> dict[str, Any]:
    from kvresearch.pipeline import _math_eval

    arm_task_scores: dict[str, dict[str, float]] = {"candidate": {}, "control": {}}
    for task in tasks:
        arm_paths = {
            "candidate": attempt_dir / "raw" / f"{task}.jsonl",
            "control": attempt_dir / "arms" / task / "unmodified_vecinfer.jsonl",
        }
        row_counts: dict[str, int] = {}
        paired_identities: dict[str, list[tuple[Any, Any, Any]]] = {}
        for arm, raw_path in arm_paths.items():
            if not raw_path.is_file():
                raise RuntimeError(f"formal 30% gate lacks fresh paired {arm} rows for {task}")
            if arm == "control":
                stock_errors = validate_stock_control_file(config, raw_path)
                if stock_errors:
                    raise RuntimeError("; ".join(stock_errors))
            raw_rows = read_jsonl(raw_path)
            row_counts[arm] = len(raw_rows)
            paired_identities[arm] = [
                (
                    int(row.get("sample_idx", index)),
                    row.get("_fixed_sample"),
                    row.get("fixed_workload_fingerprint"),
                )
                for index, row in enumerate(raw_rows)
            ]
            if task in {"gsm8k", "math500"}:
                evaluated, _ = _math_eval(
                    config, attempt_dir, task, raw_path, f"gate30_{arm}"
                )
                values = []
                for row in evaluated:
                    score = row.get("score", [False])
                    values.append(
                        100.0 if bool(score[0] if isinstance(score, list) else score) else 0.0
                    )
            else:
                values = [float(row.get("score", 0.0)) for row in raw_rows]
            if not values or len(values) != len(raw_rows):
                raise RuntimeError(f"formal 30% gate has incomplete {arm} scores for {task}")
            arm_task_scores[arm][task] = sum(values) / len(values)
        if row_counts["candidate"] != row_counts["control"]:
            raise RuntimeError(f"formal 30% paired row-count mismatch for {task}")
        if (
            paired_identities["candidate"] != paired_identities["control"]
            or any(fixed is None or fingerprint is None for _, fixed, fingerprint in paired_identities["candidate"])
        ):
            raise RuntimeError(f"formal 30% paired identity/fingerprint mismatch for {task}")
    long_tasks = [task for task in tasks if task not in {"gsm8k", "math500"}]
    aggregates = {
        arm: {
            "longbench": sum(scores[task] for task in long_tasks) / len(long_tasks),
            "gsm8k": scores["gsm8k"],
            "math500": scores["math500"],
        }
        for arm, scores in arm_task_scores.items()
    }
    candidate = aggregates["candidate"]
    baseline = aggregates["control"]
    deltas = {key: candidate[key] - baseline[key] for key in candidate}
    no_regressions = all(delta >= 0.0 for delta in deltas.values())
    any_strict_gain = any(delta > 0.0 for delta in deltas.values())
    return {"candidate": candidate, "baseline": baseline, "deltas": deltas,
            "comparison_set": "fresh_paired_control_2048_math_quality",
            "no_regressions": no_regressions, "any_strict_gain": any_strict_gain,
            "qualified_for_remaining_70pct": no_regressions and any_strict_gain}


def execute(config: Config, run_dir: Path, cycle: str, attempt: str) -> dict[str, Any]:
    cycle_dir = run_dir / "cycles" / cycle
    attempt_dir = cycle_dir / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    plan_path = cycle_dir / "experiment_plan.json"
    implementation_path = cycle_dir / "implementation_manifest.json"
    preflight_path = attempt_dir / "preflight.json"
    plan = read_json(plan_path)
    preflight = read_json(preflight_path)
    if not preflight.get("requested_resources_available"):
        raise RuntimeError("preflight did not approve requested resources")
    if plan.get("execution", {}).get("adapter", "vecinfer") != "vecinfer":
        raise ValueError("only the vecinfer execution adapter is supported")
    adapter = VecInferAdapter(config)
    approved_gpus = preflight.get("approved_gpus") or [{"index": preflight["gpu_index"], "uuid": preflight["gpu_uuid"]}]
    commands = adapter.commands(run_dir, cycle, attempt, approved_gpus, plan)
    formal = plan.get("evaluation_tier") == "formal"
    remaining_commands = adapter.commands(run_dir, cycle, attempt, approved_gpus, plan,
                                          tier_override="formal_remaining", raw_subdir="raw_remaining") if formal else []
    all_declared_commands = commands + remaining_commands
    # Sixty rows keeps enough units to refill all eight cards while avoiding
    # excessive 8B-model cold starts in the current subprocess integration.
    chunk_size = int(config.section("evaluation").get("formal_sample_chunk_size", 60))
    math_chunk_size = int(config.section("evaluation").get("formal_math_sample_chunk_size", 30))
    task_chunk_sizes = {
        str(task): math_chunk_size
        for task in config.section("evaluation").get("math_tasks", [])
    }
    effective_batch_attempts = _effective_batch_attempts(
        [int(value) for value in config.section("evaluation").get("formal_batch_sizes", [1])],
        int(config.section("evaluation").get("formal_request_concurrency", 4)),
    ) if formal else []
    sample_blocks = (
        _sample_shard_registry(commands, "initial_30pct", chunk_size, task_chunk_sizes)
        + _sample_shard_registry(remaining_commands, "remaining_70pct", chunk_size, task_chunk_sizes)
        if formal else []
    )
    dataset_marker = (
        Path(commands[0].env["KVRESEARCH_DATASETS_ROOT"]) / ".complete.json"
        if formal and commands else None
    )
    remaining_dataset_marker = (
        Path(remaining_commands[0].env["KVRESEARCH_DATASETS_ROOT"]) / ".complete.json"
        if formal and remaining_commands else None
    )
    manifest_path = attempt_dir / "run_manifest.json"
    manifest = control_envelope(
        "attempt_run_manifest",
        run_dir.name,
        "kvresearch.executor",
        [path_ref(plan_path), path_ref(implementation_path), path_ref(preflight_path)],
        command=[{"name": cmd.name, "argv": cmd.argv, "cwd": str(cmd.cwd), "gpu_index": cmd.gpu_index,
                  "gpu_uuid": cmd.gpu_uuid,
                  "phase": "remaining_70pct" if cmd in remaining_commands else "initial_30pct" if formal else "development"}
                 for cmd in all_declared_commands],
        cwd=str(Path(plan["workspace"])),
        env_allowlist=sorted({key for cmd in all_declared_commands for key in cmd.env}),
        input_hashes={
            "plan": sha256_file(plan_path),
            "implementation": sha256_file(implementation_path),
            "preflight": sha256_file(preflight_path),
            "host_executor": sha256_file(Path(__file__)),
            "host_remote": sha256_file(Path(__file__).resolve().with_name("remote.py")),
            "host_config": sha256_file(config.path),
            **({"initial_dataset_marker": sha256_file(dataset_marker)} if dataset_marker else {}),
            **({"remaining_dataset_marker": sha256_file(remaining_dataset_marker)} if remaining_dataset_marker else {}),
        },
        seeds=[config.section("evaluation")["seed"]],
        shards=[{"gpu_index": gpu["index"], "gpu_uuid": gpu.get("uuid"),
                 "tasks": [], "assignment": "dynamic_shared_queue"}
                for gpu in approved_gpus],
        scheduling={"mode": "dynamic_sample_shard_queue" if formal else "dynamic_shared_queue",
                    "gpu_worker_count": min(len(approved_gpus), len(commands)),
                    "workers_per_gpu": 1,
                    "intra_gpu_request_concurrency": int(config.section("evaluation").get("formal_request_concurrency", 4)) if formal else 1,
                    "effective_batch_attempts": effective_batch_attempts if formal else None,
                    "sample_chunk_size": chunk_size if formal else None,
                    "math_sample_chunk_size": math_chunk_size if formal else None,
                    "deterministic_merge": "global_sample_idx_atomic" if formal else None},
        sample_blocks=sample_blocks,
        started_at=utcnow(),
        timeout=max((cmd.timeout_seconds for cmd in commands), default=0),
        timeout_scope="per_sample_block" if formal else "per_task",
        attempt_deadline_seconds=None,
        pid=os.getpid(),
        status="partial",
    )
    atomic_write_json(manifest_path, manifest)
    started = time.monotonic()
    batch_sizes = [int(value) for value in config.section("evaluation").get("formal_batch_sizes", [1])] if formal else []
    if formal:
        batch_sizes = _effective_batch_attempts(
            batch_sizes,
            int(config.section("evaluation").get("formal_request_concurrency", 4)),
        )
    if formal:
        records = _run_sample_sharded_gpu_pool(
            adapter, commands, approved_gpus, attempt_dir, batch_sizes,
            "initial_30pct",
            chunk_size,
            task_chunk_sizes,
        )
    else:
        records = _run_dynamic_gpu_pool(
            adapter, commands, approved_gpus, attempt_dir, batch_sizes, "development"
        )
    primary_ok = len([record for record in records if record["returncode"] == 0]) == len(commands)
    task_names = [cmd.name.removeprefix("generate-") for cmd in commands]
    stock_control_validation = validate_attempt_stock_controls(config, attempt_dir, task_names)
    if primary_ok and not stock_control_validation["valid"]:
        primary_ok = False
    diagnostic_validation = validate_diagnostic_outputs(plan, attempt_dir)
    if primary_ok and diagnostic_validation["required"] and not diagnostic_validation["valid"]:
        primary_ok = False
    gate = None
    remaining_diagnostic_validation = None
    if formal and primary_ok:
        try:
            gate = _formal_30pct_gate(config, attempt_dir, [cmd.name.removeprefix("generate-") for cmd in commands])
            atomic_write_json(attempt_dir / "formal_30pct_gate.json", gate)
        except Exception as exc:
            gate = {"qualified_for_remaining_70pct": False, "evaluation_error": str(exc)}
            atomic_write_json(attempt_dir / "formal_30pct_gate.json", gate)
            primary_ok = False
    if formal and primary_ok and gate and gate["qualified_for_remaining_70pct"]:
        records.extend(_run_sample_sharded_gpu_pool(
            adapter, remaining_commands, approved_gpus, attempt_dir, batch_sizes,
            "remaining_70pct",
            chunk_size,
            task_chunk_sizes,
        ))
        remaining_stock = validate_attempt_stock_controls(
            config, attempt_dir,
            [cmd.name.removeprefix("generate-") for cmd in remaining_commands],
            remaining=True,
        )
        stock_control_validation = {
            "valid": stock_control_validation["valid"] and remaining_stock["valid"],
            "errors": stock_control_validation["errors"] + remaining_stock["errors"],
            "paths": stock_control_validation["paths"] + remaining_stock["paths"],
        }
        if not remaining_stock["valid"]:
            primary_ok = False
        remaining_diagnostic_validation = validate_diagnostic_outputs(
            plan, attempt_dir, remaining=True
        )
        if (
            remaining_diagnostic_validation["required"]
            and not remaining_diagnostic_validation["valid"]
        ):
            primary_ok = False
    exit_codes = {f"{record['phase']}:{record['task']}": int(record["returncode"]) for record in records}
    timed_out = any(record["timed_out"] for record in records)
    cleanup_failed = any(record.get("cleanup_failed", False) for record in records)
    oom = any(record["oom"] and record["returncode"] != 0 for record in records)
    completed = [f"{record['phase']}:{record['task']}" for record in records if record["returncode"] == 0]
    runtime_trace = [sample for record in records for sample in record.get("runtime_trace", [])]
    atomic_write_jsonl(attempt_dir / "runtime_trace.jsonl", runtime_trace)
    scheduler_journal = attempt_dir / "scheduler_journal.jsonl"
    expected = [f"{'initial_30pct' if formal else 'development'}:{cmd.name.removeprefix('generate-')}" for cmd in commands]
    if formal and gate and gate.get("qualified_for_remaining_70pct"):
        expected.extend(f"remaining_70pct:{cmd.name.removeprefix('generate-')}" for cmd in remaining_commands)
    diagnostic_failed = (
        diagnostic_validation["required"] and not diagnostic_validation["valid"]
    ) or bool(
        remaining_diagnostic_validation
        and remaining_diagnostic_validation["required"]
        and not remaining_diagnostic_validation["valid"]
    )
    overall = 0 if primary_ok and len(completed) == len(expected) else next((code for code in exit_codes.values() if code != 0), 1)
    if overall == 0:
        error_code = None
    elif cleanup_failed:
        error_code = "E_REMOTE_CLEANUP"
    elif oom:
        error_code = "E_GPU_OOM"
    elif timed_out:
        error_code = "E_TIMEOUT"
    elif diagnostic_failed:
        error_code = "E_DIAGNOSTICS_INCOMPLETE"
    elif not stock_control_validation["valid"]:
        error_code = "E_STOCK_CONTROL_INVALID"
    else:
        error_code = "E_EXECUTION"
    exit_artifact = control_envelope(
        "attempt_exit",
        run_dir.name,
        "kvresearch.executor",
        [path_ref(manifest_path)],
        exit_code=overall,
        command_exit_codes=exit_codes,
        ended_at=utcnow(),
        elapsed_seconds=time.monotonic() - started,
        timed_out=timed_out,
        remote_cleanup_confirmed=not cleanup_failed,
        oom=oom,
        completed_tasks=completed,
        missing_tasks=[task for task in expected if task not in completed],
        log_hashes={str(path.relative_to(attempt_dir)): sha256_file(path) for path in sorted((attempt_dir / "logs").glob("*.log"))},
        scheduler_journal=(
            {"path": str(scheduler_journal), "sha256": sha256_file(scheduler_journal)}
            if scheduler_journal.is_file() else None
        ),
        task_executions=[{key: value for key, value in record.items() if key != "runtime_trace"} for record in records],
        diagnostic_validation=diagnostic_validation,
        remaining_diagnostic_validation=remaining_diagnostic_validation,
        stock_control_validation=stock_control_validation,
        formal_30pct_gate=gate,
        status="complete" if overall == 0 else "failed",
        error_code=error_code,
    )
    atomic_write_json(attempt_dir / "exit.json", exit_artifact)
    return exit_artifact


def monitor(config: Config, run_dir: Path, cycle: str, attempt: str) -> dict[str, Any]:
    attempt_dir = run_dir / "cycles" / cycle / "attempts" / attempt
    exit_path = attempt_dir / "exit.json"
    if exit_path.is_file():
        return read_json(exit_path)
    manifest = read_json(attempt_dir / "run_manifest.json")
    pid = int(manifest.get("pid", -1))
    alive = False
    if pid > 0:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            pass
    if alive:
        return {"alive": True, "pid": pid, "status": "running"}
    cleanup_confirmed = RemotePod(config).cleanup_registry(
        attempt_dir / "remote_processes"
    )
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    artifact = control_envelope(
        "attempt_exit", run_dir.name, "kvresearch.executor.monitor", [path_ref(attempt_dir / "run_manifest.json")],
        exit_code=1, command_exit_codes={}, ended_at=utcnow(), elapsed_seconds=None,
        timed_out=False, oom=False, completed_tasks=[], missing_tasks=[],
        log_hashes={
            key: sha256_file(path) for key, path in (("stdout", stdout_path), ("stderr", stderr_path)) if path.is_file()
        },
        status="failed",
        error_code="E_REMOTE_ENV" if cleanup_confirmed else "E_REMOTE_CLEANUP",
        remote_cleanup_confirmed=cleanup_confirmed,
        interruption="executor process is no longer alive",
    )
    atomic_write_json(exit_path, artifact)
    return artifact
