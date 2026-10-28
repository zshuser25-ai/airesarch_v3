from __future__ import annotations

import csv
import io
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from kvresearch.config import Config
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, path_ref, utcnow


class RemoteCleanupError(RuntimeError):
    """A remote process group could not be proven terminated."""


class RemotePod:
    def __init__(self, config: Config):
        self.config = config
        self.remote = config.section("remote")

    def ssh_argv(self, command: str) -> list[str]:
        argv = ["ssh"]
        socket = Path(self.remote.get("control_socket", ""))
        if str(socket) and socket.exists():
            argv.extend(["-S", str(socket)])
        argv.extend(
            [
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                "-p", str(self.remote["port"]),
                f"{self.remote['user']}@{self.remote['host']}",
                command,
            ]
        )
        return argv

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.ssh_argv(command), text=True, capture_output=True, timeout=timeout, check=False)

    def runtime_sample(self, gpu_uuid: str, phase: str) -> dict[str, Any]:
        command = (
            "nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits; "
            "awk '/MemTotal:|MemAvailable:/{print $1,$2}' /proc/meminfo"
        )
        result = self.run(command)
        if result.returncode != 0:
            return {"captured_at": utcnow(), "gpu_uuid": gpu_uuid, "gpu_memory_used": None,
                    "gpu_utilization": None, "cpu_memory_used": None, "phase": phase,
                    "sampling_error": result.stderr.strip()}
        gpu_memory = gpu_util = None
        mem_total = mem_available = None
        for line in result.stdout.splitlines():
            if line.startswith("GPU-"):
                values = [item.strip() for item in line.split(",")]
                if values[0] == gpu_uuid and len(values) >= 3:
                    gpu_memory, gpu_util = int(values[1]), int(values[2])
            elif line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
        return {
            "captured_at": utcnow(), "gpu_uuid": gpu_uuid,
            "gpu_memory_used": gpu_memory, "gpu_utilization": gpu_util,
            "cpu_memory_used": (mem_total - mem_available) if mem_total is not None and mem_available is not None else None,
            "phase": phase,
        }

    def environment_snapshot(self) -> dict[str, Any]:
        gpu_result = self.run(
            "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu,driver_version "
            "--format=csv,noheader,nounits"
        )
        runtime_paths = ":".join(str(item) for item in self.remote.get("runtime_pythonpath", []))
        probe = (
            "hostname; uname -srmo; "
            f"PYTHONPATH={shlex.quote(runtime_paths)} {shlex.quote(self.remote['vecinfer_python'])} -c "
            + shlex.quote(
                "import json,platform; import torch,transformers,datasets,jieba,rouge,fuzzywuzzy,pykeops,faiss; "
                "import bindings,pq_bindings; print(json.dumps({'python':platform.python_version(),"
                "'torch':torch.__version__,'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,"
                "'datasets':datasets.__version__,'pykeops':pykeops.__version__,'bindings':bindings.__file__,"
                "'pq_bindings':pq_bindings.__file__}))"
            )
            + f"; df -Pk {shlex.quote(str(self.config.project_root))} | tail -n 1"
        )
        info_result = self.run(probe)
        info_lines = info_result.stdout.splitlines()
        software: dict[str, Any] = {}
        for line in info_lines:
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "python" in parsed:
                    software = parsed
            except json.JSONDecodeError:
                pass
        gpus = []
        for row in csv.reader(io.StringIO(gpu_result.stdout)):
            values = [item.strip() for item in row]
            if len(values) >= 7:
                gpus.append({"index": int(values[0]), "name": values[1], "uuid": values[2],
                             "memory_total_mib": int(values[3]), "memory_free_mib": int(values[4]),
                             "utilization_percent": int(values[5]), "driver_version": values[6]})
        hostname = info_lines[0].strip() if info_lines else "unknown"
        valid = gpu_result.returncode == 0 and info_result.returncode == 0 and bool(software) and (
            not self.remote.get("expected_hostname") or hostname == self.remote["expected_hostname"]
        )
        return {
            "schema_version": 1, "captured_at": utcnow(), "host": hostname,
            "kernel": info_lines[1].strip() if len(info_lines) > 1 else "unknown",
            "connection": {"host": self.remote["host"], "port": self.remote["port"], "user": self.remote["user"]},
            "gpus": gpus, "software": software,
            "runtime_pythonpath": self.remote.get("runtime_pythonpath", []),
            "filesystem": next((line for line in info_lines if line.split() and line.split()[-1].startswith("/nfs/")), "unknown"),
            "constraints": {
                "expected_hostname": self.remote.get("expected_hostname"),
                "minimum_free_gpu_memory_mib": self.remote.get("minimum_free_gpu_memory_mib"),
                "maximum_idle_gpu_utilization_percent": self.remote.get("maximum_idle_gpu_utilization_percent"),
            },
            "valid": valid,
            "errors": [item for item in (gpu_result.stderr.strip(), info_result.stderr.strip()) if item],
        }

    def preflight(self, run_dir: Path, cycle: str, attempt: str) -> dict[str, Any]:
        stale_records = sorted(
            (run_dir / "cycles").glob("*/attempts/*/remote_processes/*.pid")
        )
        failed_cleanup = [
            str(record) for record in stale_records
            if not self._cleanup_execution(record.stem, str(record))
        ]
        if failed_cleanup:
            raise RuntimeError(
                "E_REMOTE_CLEANUP: stale remote execution groups remain: "
                + ", ".join(failed_cleanup)
            )
        query = (
            "nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu,pstate,driver_version --format=csv,noheader,nounits"
        )
        gpu_result = self.run(query)
        if gpu_result.returncode != 0:
            raise RuntimeError(f"remote GPU query failed: {gpu_result.stderr.strip()}")
        rows = list(csv.reader(io.StringIO(gpu_result.stdout)))
        gpus: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 10:
                continue
            values = [item.strip() for item in row]
            gpus.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "uuid": values[2],
                    "memory_total_mib": int(values[3]),
                    "memory_used_mib": int(values[4]),
                    "memory_free_mib": int(values[5]),
                    "utilization_percent": int(values[6]),
                    "temperature_c": int(values[7]),
                    "pstate": values[8],
                    "driver_version": values[9],
                }
            )
        if not gpus:
            raise RuntimeError("no GPUs returned by remote preflight")
        min_free = int(self.remote.get("minimum_free_gpu_memory_mib", 0))
        max_util = int(self.remote.get("maximum_idle_gpu_utilization_percent", 100))
        eligible = [gpu for gpu in gpus if gpu["memory_free_mib"] >= min_free and gpu["utilization_percent"] <= max_util]
        eligible.sort(key=lambda gpu: (-gpu["memory_free_mib"], gpu["index"]))
        plan_path = run_dir / "cycles" / cycle / "experiment_plan.json"
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        formal = plan_data.get("evaluation_tier") == "formal"
        default_gpu_count = int(self.config.section("evaluation").get("formal_gpu_count", 8)) if formal else 1
        requested_count = int(plan_data.get("execution", {}).get("gpu_count", default_gpu_count))
        if requested_count < 1:
            raise ValueError(f"plan execution.gpu_count must be positive, got {requested_count}")
        selected_gpus = eligible[:requested_count]
        selected = selected_gpus[0] if selected_gpus else max(gpus, key=lambda gpu: (gpu["memory_free_mib"], -gpu["utilization_percent"]))
        available = len(selected_gpus) == requested_count

        runtime_paths = ":".join(str(item) for item in self.remote.get("runtime_pythonpath", []))
        runtime_probe = (
            f"PYTHONPATH={shlex.quote(runtime_paths)} {shlex.quote(self.remote['vecinfer_python'])} -c "
            + shlex.quote(
                "import json; import torch,transformers,datasets,jieba,rouge,fuzzywuzzy,pykeops,faiss; "
                "import bindings,pq_bindings; print(json.dumps({'runtime_imports':True,"
                "'runtime_torch':torch.__version__,'runtime_transformers':transformers.__version__,"
                "'runtime_datasets':datasets.__version__,'runtime_pykeops':pykeops.__version__,"
                "'bindings':bindings.__file__}))"
            )
        )
        info_command = (
            "hostname; uname -srmo; "
            f"{shlex.quote(self.remote['python'])} -c "
            + shlex.quote(
                "import json,platform; import torch,transformers,datasets; "
                "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,"
                "'datasets':datasets.__version__}))"
            )
            + f"; {runtime_probe}; df -Pk {shlex.quote(str(self.config.project_root))} | tail -n 1; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits"
        )
        info_result = self.run(info_command)
        info_lines = info_result.stdout.splitlines()
        software: dict[str, Any] = {}
        for line in info_lines:
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and "python" in parsed:
                    software.update(parsed)
                elif isinstance(parsed, dict) and "runtime_imports" in parsed:
                    software.update(parsed)
            except json.JSONDecodeError:
                pass
        processes = [line for line in info_lines if line.startswith("GPU-")]
        run_id = run_dir.name
        plan = plan_path
        hostname_ok = not self.remote.get("expected_hostname") or (
            bool(info_lines) and info_lines[0].strip() == self.remote["expected_hostname"]
        )
        approved = available and info_result.returncode == 0 and software.get("runtime_imports") is True and hostname_ok
        artifact = control_envelope(
            "preflight",
            run_id,
            "kvresearch.remote",
            [path_ref(plan)],
            host=info_lines[0].strip() if info_lines else "unknown",
            captured_at=utcnow(),
            gpu_index=selected["index"],
            gpu_uuid=selected["uuid"],
            gpu_memory={"total_mib": selected["memory_total_mib"], "used_mib": selected["memory_used_mib"], "free_mib": selected["memory_free_mib"]},
            gpu_load={"utilization_percent": selected["utilization_percent"], "temperature_c": selected["temperature_c"], "pstate": selected["pstate"]},
            processes=processes,
            software=software,
            disk={"raw": next((line for line in info_lines if line.split() and line.split()[-1].startswith("/nfs/")), "unknown")},
            all_gpus=gpus,
            approved_gpus=selected_gpus,
            requested_gpu_count=requested_count,
            requested_resources_available=approved,
            status="complete" if approved else "failed",
            error_code=None if approved else "E_REMOTE_ENV",
            probe_returncode=info_result.returncode,
            probe_stderr=info_result.stderr.strip(),
        )
        output = run_dir / "cycles" / cycle / "attempts" / attempt / "preflight.json"
        atomic_write_json(output, artifact)
        return artifact

    def execute_argv(self, argv: list[str], cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
        execution_env = dict(env)
        registry_dir = execution_env.pop("KVRESEARCH_REMOTE_REGISTRY_DIR", "")
        execution_unit = execution_env.pop("KVRESEARCH_REMOTE_EXECUTION_UNIT", "unknown")
        token = uuid.uuid4().hex
        execution_env["KVRESEARCH_EXECUTION_TOKEN"] = token
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(execution_env.items())
        )
        payload = f"cd {shlex.quote(str(cwd))} && "
        if assignments:
            payload += assignments + " "
        payload += shlex.join(argv)

        # An ssh client killed by subprocess.run(timeout=...) does not reliably
        # kill the command that sshd already started.  Put every execution in a
        # uniquely named remote session and persist its process-group id before
        # waiting.  The remote trap handles ordinary disconnects; the explicit
        # follow-up command makes timeout cleanup deterministic and narrowly
        # scoped to this invocation.
        token_label = f"kvresearch-token-{token}"
        marker_root = Path(registry_dir) if registry_dir else Path("/tmp")
        marker = str(marker_root / f"{token}.pid")
        cleanup_body = (
            'if [ -n "${child:-}" ]; then '
            'kill -TERM -- "-$child" 2>/dev/null || true; '
            'for delay in 1 2 3 4 5; do '
            'kill -0 "$child" 2>/dev/null || break; sleep 1; '
            'done; '
            'kill -KILL -- "-$child" 2>/dev/null || true; '
            'fi; '
            f'rm -f {shlex.quote(marker)}'
        )
        wrapped = (
            f"umask 022; mkdir -p {shlex.quote(str(marker_root))}; "
            f"chmod 0755 {shlex.quote(str(marker_root))}; child=''; "
            f"trap {shlex.quote(cleanup_body)} HUP INT TERM EXIT; "
            f"setsid env KVRESEARCH_EXECUTION_TOKEN={shlex.quote(token)} "
            f"timeout --signal=TERM --kill-after=10 {int(timeout)} sh -c "
            + shlex.quote('eval "$1"; rc=$?; exit "$rc"')
            + f" {shlex.quote(token_label)} {shlex.quote(payload)} & child=$!; "
            'start=$(awk \'{print $22}\' "/proc/$child/stat" 2>/dev/null || true); '
            f"printf '%s %s %s %s\\n' {shlex.quote(token)} \"$child\" \"$start\" "
            f"{shlex.quote(execution_unit)} > {shlex.quote(marker)}; "
            f"chmod 0600 {shlex.quote(marker)}; "
            'wait "$child"; rc=$?; '
            "trap - HUP INT TERM EXIT; "
            f"rm -f {shlex.quote(marker)}; exit \"$rc\""
        )
        try:
            # Deadline enforcement happens remotely so an abrupt host death
            # still has a bounded lifetime.  The local timeout is a transport
            # backstop and deliberately gives remote cleanup time to finish.
            return self.run(wrapped, timeout=timeout + 30)
        except subprocess.TimeoutExpired as exc:
            if not self._cleanup_execution(token, marker):
                raise RemoteCleanupError(
                    f"E_REMOTE_CLEANUP: could not confirm termination for {execution_unit}"
                ) from exc
            raise

    def _cleanup_execution(self, token: str, marker: str) -> bool:
        """Kill only the timed-out remote process group identified by token."""
        token_env = f"KVRESEARCH_EXECUTION_TOKEN={token}"
        cleanup = (
            f"marker={shlex.quote(marker)}; expected={shlex.quote(token)}; "
            'if [ -r "$marker" ]; then '
            "read actual pgid expected_start unit < \"$marker\" || true; "
            'case "$pgid" in ""|*[!0-9]*) pgid="";; esac; '
            'if [ "$actual" != "$expected" ] || [ -z "$pgid" ] || [ -z "$expected_start" ]; then '
            'echo CLEANUP_FAILED; exit 1; fi; '
            'if [ -r "/proc/$pgid/stat" ]; then '
            'current_start=$(awk \'{print $22}\' "/proc/$pgid/stat" 2>/dev/null || true); '
            'if [ "$current_start" != "$expected_start" ]; then echo CLEANUP_FAILED; exit 1; fi; fi; '
            'members=0; matched=0; foreign=0; '
            'if [ "$actual" = "$expected" ] && [ -n "$pgid" ]; then '
            'for proc in /proc/[0-9]*; do '
            'member_pgid=$(awk \'{print $5}\' "$proc/stat" 2>/dev/null || true); '
            'if [ "$member_pgid" = "$pgid" ]; then members=$((members+1)); '
            f"if tr '\\000' '\\n' < \"$proc/environ\" 2>/dev/null | grep -Fx {shlex.quote(token_env)} >/dev/null; "
            'then matched=$((matched+1)); else foreign=$((foreign+1)); fi; fi; done; '
            'if [ "$members" -gt 0 ] && [ "$matched" -gt 0 ] && [ "$foreign" -eq 0 ]; then '
            'kill -TERM -- "-$pgid" 2>/dev/null || true; '
            'for delay in 1 2 3 4 5; do '
            'kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; '
            'done; '
            'kill -KILL -- "-$pgid" 2>/dev/null || true; '
            'for delay in 1 2 3 4 5; do '
            'kill -0 -- "-$pgid" 2>/dev/null || break; sleep 1; '
            'done; '
            'fi; fi; '
            'if [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null; then '
            "echo CLEANUP_FAILED; exit 1; "
            'fi; rm -f "$marker"; echo CLEANUP_CONFIRMED; '
            "else "
            'found=0; for proc in /proc/[0-9]*; do '
            f"if tr '\\000' '\\n' < \"$proc/environ\" 2>/dev/null | grep -Fx {shlex.quote(token_env)} >/dev/null; "
            'then found=1; break; fi; done; if [ "$found" -ne 0 ]; then '
            "echo CLEANUP_FAILED; exit 1; "
            "fi; echo CLEANUP_CONFIRMED; fi"
        )
        try:
            result = self.run(cleanup, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "CLEANUP_CONFIRMED" in result.stdout

    def cleanup_registry(self, registry_dir: Path) -> bool:
        """Reap all durable process-group records belonging to one attempt."""
        records = sorted(registry_dir.glob("*.pid")) if registry_dir.is_dir() else []
        confirmed = True
        for record in records:
            if not self._cleanup_execution(record.stem, str(record)):
                confirmed = False
        return confirmed
