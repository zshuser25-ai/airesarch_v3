from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kvresearch import __version__
from kvresearch.config import Config
from kvresearch.schema import control_envelope
from kvresearch.stock_control import stock_codebook_manifest
from kvresearch.util import atomic_write_json, atomic_write_text, path_ref, sha256_file, sha256_tree


def _git_commit(repo: Path) -> str | None:
    git_dir = repo / ".git"
    if not git_dir.exists():
        return None
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _model_identity(model_dir: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    files = [path_ref(model_dir / name) for name in names if (model_dir / name).is_file()]
    return {"path": str(model_dir), "identity_files": files}


def build_inputs_lock(config: Config, run_id: str, request_path: Path) -> dict[str, Any]:
    project = config.section("project")
    model = config.section("model")
    baseline = config.section("baseline")
    dataset_locks: dict[str, Any] = {}
    manifest_hashes: dict[str, str] = {}
    for tier, setting in (("development", "fixed_testset"), ("formal", "formal_testset"),
                          ("formal_remaining", "formal_remaining_testset")):
        dataset_dir = Path(project[setting])
        manifest = dataset_dir / "manifest.json"
        data = __import__("json").loads(manifest.read_text(encoding="utf-8"))
        dataset_locks[tier] = {
            "manifest": path_ref(manifest),
            "tasks": {
                task: {"path": str((dataset_dir / entry["output"]["path"]).resolve()),
                       "sha256": entry["output"]["sha256"], "rows": entry["output"]["rows"]}
                for task, entry in data["tasks"].items()
            },
        }
        manifest_hashes[f"{tier}_manifest"] = sha256_file(manifest)
    source = Path(baseline["source"])
    source_hash = sha256_tree(
        source / "scripts",
        suffixes=(".py", ".sh", ".json"),
        excluded=("__pycache__", "build", "dist"),
    )
    environment_path = config.project_root / "environment" / "current_pod.json"
    inputs = [path_ref(request_path), path_ref(config.path), path_ref(config.catalog_path)]
    formal_baseline = Path(project["formal_baseline"])
    inputs.append(path_ref(formal_baseline))
    environment = path_ref(environment_path) if environment_path.is_file() else None
    if environment:
        inputs.append(environment)
    return control_envelope(
        "inputs_lock",
        run_id,
        "kvresearch.initialize",
        inputs,
        model=_model_identity(Path(model["path"])),
        baselines={
            "report": path_ref(Path(project["baseline_report"])),
            "formal_30pct": path_ref(formal_baseline),
            "source": {
                "path": str(source),
                "git_commit": _git_commit(source),
                "scripts_sha256": source_hash,
                "read_only": True,
            },
            "secondary_source": {
                "path": baseline["secondary_source"],
                "git_commit": _git_commit(Path(baseline["secondary_source"])),
                "read_only": True,
            },
            "stock_vecinfer_control": {
                **stock_codebook_manifest(config),
                "assignment_policy": "squared_euclidean",
                "read_only": True,
            },
        },
        datasets=dataset_locks,
        evaluation=config.section("evaluation"),
        cooldown=config.section("cooldown"),
        software={
            "python_executable": config.section("remote")["python"],
            "vecinfer_python_executable": config.section("remote")["vecinfer_python"],
            "runtime_pythonpath": config.section("remote").get("runtime_pythonpath", []),
        },
        environment=environment,
        hashes={
            "config": sha256_file(config.path),
            "catalog": sha256_file(config.catalog_path),
            **manifest_hashes,
            "baseline_scripts": source_hash,
            "framework_source": sha256_tree(
                config.project_root / "kvresearch",
                suffixes=(".py", ".json", ".md"),
                excluded=("__pycache__",),
            ),
        },
    )


def initialize_run(config: Config, request: str, run_id: str | None = None, parent_run_id: str | None = None) -> Path:
    if not request.strip():
        raise ValueError("research request must not be empty")
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"kv1bit-{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id contains unsupported characters")
    run_dir = config.runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    request_path = run_dir / "request.md"
    atomic_write_text(request_path, request.rstrip() + "\n")
    manifest = control_envelope(
        "run_manifest",
        run_id,
        "kvresearch.initialize",
        [path_ref(request_path), path_ref(config.path), path_ref(config.catalog_path)],
        parent_run_id=parent_run_id,
        framework_version=__version__,
        project_root=str(config.project_root),
        catalog_sha256=sha256_file(config.catalog_path),
        config_path=str(config.path),
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "inputs.lock.json", build_inputs_lock(config, run_id, request_path))
    return run_dir
