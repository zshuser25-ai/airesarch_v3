from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kvresearch.config import Config
from kvresearch.util import read_jsonl, sha256_file


STOCK_ARM_ROLE = "stock_vecinfer_control"
STOCK_ASSIGNMENT_POLICY = "squared_euclidean"


def stock_codebook_manifest(config: Config) -> dict[str, Any]:
    root = Path(config.section("baseline")["centroids"]).resolve()
    storage = config.section("storage")["vecinfer"]
    layers = int(storage["layers"])
    key_m = int(storage["key_subspaces"])
    key_bits = int(storage["key_bits_per_code"])
    value_m = int(storage["value_subspaces"])
    value_bits = int(storage["value_bits_per_code"])
    paths = []
    for layer in range(layers):
        paths.extend((
            root / f"key_cent_layer{layer}_{key_m}_{key_bits}.pq.pt",
            root / f"val_cent_layer{layer}_{value_m}_{value_bits}.pq.pt",
        ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"stock VecInfer codebooks incomplete: {missing[:3]}")
    files = [{"path": path.name, "sha256": sha256_file(path)} for path in paths]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "sha256": digest, "files": files, "file_count": len(files)}


def stock_control_env(config: Config) -> dict[str, str]:
    manifest = stock_codebook_manifest(config)
    return {
        "KVRESEARCH_REQUIRE_STOCK_VECINFER_CONTROL": "1",
        "VECINFER_STOCK_CODEBOOK_ROOT": manifest["root"],
        "VECINFER_STOCK_CODEBOOK_MANIFEST_SHA256": manifest["sha256"],
        "VECINFER_STOCK_ASSIGNMENT_POLICY": STOCK_ASSIGNMENT_POLICY,
    }


def validate_stock_control_file(config: Config, path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"E_STOCK_CONTROL_MISSING: {path}"]
    try:
        rows = read_jsonl(path)
        manifest = stock_codebook_manifest(config)
    except Exception as exc:
        return [f"E_STOCK_CONTROL_INVALID: {exc}"]
    if not rows:
        return [f"E_STOCK_CONTROL_INVALID: empty control file {path}"]
    for index, row in enumerate(rows, 1):
        prefix = f"E_STOCK_CONTROL_INVALID: {path}:line {index}"
        if row.get("arm_role") != STOCK_ARM_ROLE:
            errors.append(f"{prefix} arm_role must be {STOCK_ARM_ROLE}")
        if row.get("stock_vecinfer_codebook_root") != manifest["root"]:
            errors.append(f"{prefix} stock codebook root mismatch")
        if row.get("stock_vecinfer_codebook_manifest_sha256") != manifest["sha256"]:
            errors.append(f"{prefix} stock codebook manifest hash mismatch")
        if row.get("stock_vecinfer_assignment_policy") != STOCK_ASSIGNMENT_POLICY:
            errors.append(f"{prefix} assignment policy must be {STOCK_ASSIGNMENT_POLICY}")
        if row.get("candidate_codebook_root") not in (None, ""):
            errors.append(f"{prefix} control loaded a candidate codebook root")
        if row.get("candidate_loader") not in (None, ""):
            errors.append(f"{prefix} control used a candidate loader")
    return errors


def validate_attempt_stock_controls(
    config: Config, attempt_dir: Path, tasks: list[str], *, remaining: bool = False,
) -> dict[str, Any]:
    arms_dir = "arms_remaining" if remaining else "arms"
    failures: list[str] = []
    paths = []
    for task in tasks:
        path = attempt_dir / arms_dir / task / "unmodified_vecinfer.jsonl"
        paths.append(str(path))
        failures.extend(validate_stock_control_file(config, path))
    return {"valid": not failures, "errors": failures, "paths": paths}
