from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from kvresearch.mechanisms import SELECTABLE_FAMILIES, canonical_mechanism_family, validate_candidate_architecture
from kvresearch.util import read_json, read_jsonl, sha256_file


SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "artifact.schema.json"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]


class ArtifactValidator:
    def __init__(self, catalog_path: Path):
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        self.catalog = catalog
        self.definitions = {entry["id"]: entry for entry in catalog["artifacts"]}
        self.base_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def validate_control(self, path: Path, expected_type: str | None = None, verify_inputs: bool = True) -> ValidationResult:
        errors: list[str] = []
        try:
            value = read_json(path)
        except Exception as exc:
            return ValidationResult(False, (f"E_ARTIFACT_INVALID: {exc}",))
        try:
            jsonschema.validate(value, self.base_schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"E_SCHEMA_MISMATCH: {exc.message}")

        artifact_type = expected_type or str(value.get("artifact_type", ""))
        if expected_type and value.get("artifact_type") != expected_type:
            errors.append(f"E_SCHEMA_MISMATCH: expected artifact_type={expected_type}")
        definition = self.definitions.get(artifact_type)
        if not definition:
            errors.append(f"E_SCHEMA_MISMATCH: unknown artifact_type={artifact_type}")
        else:
            for field in definition.get("required_fields", []):
                if field not in value:
                    errors.append(f"E_SCHEMA_MISMATCH: missing field {field}")
            allowed = definition.get("allowed_actions")
            if allowed and value.get("action") not in allowed:
                errors.append(f"E_SCHEMA_MISMATCH: invalid action {value.get('action')}")
        if artifact_type == "selection":
            ranking = value.get("ranking", [])
            selected = value.get("selected_candidate_id")
            if not isinstance(ranking, list) or len(ranking) != 4 or len(set(ranking)) != 4:
                errors.append("E_SCHEMA_MISMATCH: selection ranking must contain exactly 4 unique candidate IDs")
            elif ranking[0] != selected:
                errors.append("E_SCHEMA_MISMATCH: selected candidate must be first in ranking")
            candidate_ref = next((Path(ref.get("path", "")) for ref in value.get("inputs", [])
                                  if Path(ref.get("path", "")).name == "candidates.jsonl"), None)
            if candidate_ref and candidate_ref.is_file():
                candidate_rows = read_jsonl(candidate_ref)
                candidate_ids = {row.get("candidate_id") for row in candidate_rows}
                if set(ranking) != candidate_ids:
                    errors.append("E_SCHEMA_MISMATCH: selection ranking must be a permutation of candidate IDs")
                selected_row = next((row for row in candidate_rows if row.get("candidate_id") == selected), None)
                if canonical_mechanism_family(selected_row or {}) not in SELECTABLE_FAMILIES:
                    errors.append("E_ARCHITECTURE_QUOTA: selection must choose one of the three architecture candidates")
                control = next(
                    (row for row in candidate_rows if canonical_mechanism_family(row) == "stock_vecinfer_control"),
                    None,
                )
                if control and ranking and ranking[-1] != control.get("candidate_id"):
                    errors.append("E_ARCHITECTURE_QUOTA: stock VecInfer control must be last and is not selectable")

        if artifact_type == "experiment_plan":
            workspace = value.get("workspace")
            if not isinstance(workspace, str) or not workspace.strip():
                errors.append("E_SCHEMA_MISMATCH: experiment_plan workspace must be a non-empty absolute-path string")
            elif not Path(workspace).is_absolute():
                errors.append("E_SCHEMA_MISMATCH: experiment_plan workspace must be an absolute path")
            code_scope = value.get("code_scope")
            if not isinstance(code_scope, list) or not code_scope or not all(
                isinstance(item, str) and item.strip() for item in code_scope
            ):
                errors.append("E_SCHEMA_MISMATCH: experiment_plan code_scope must be a non-empty string array")
            budgets = value.get("budgets")
            if not isinstance(budgets, dict):
                errors.append("E_SCHEMA_MISMATCH: experiment_plan budgets must be an object")
            decoding = value.get("decoding")
            if not isinstance(decoding, dict) or decoding.get("math_max_total_length") != 2048:
                errors.append("E_SCHEMA_MISMATCH: experiment_plan decoding.math_max_total_length must equal 2048")
            elif isinstance(decoding.get("longbench"), dict) and "max_total_length" in decoding["longbench"]:
                errors.append(
                    "E_SCHEMA_MISMATCH: experiment_plan must not apply the 2048 math total-length cap to LongBench"
                )

        if verify_inputs:
            for ref in value.get("inputs", []):
                source = Path(ref.get("path", ""))
                if not source.is_file():
                    errors.append(f"E_INPUT_MISSING: {source}")
                elif sha256_file(source) != ref.get("sha256"):
                    errors.append(f"E_INPUT_DRIFT: {source}")
        return ValidationResult(not errors, tuple(errors))

    def validate_jsonl(self, path: Path, artifact_type: str) -> ValidationResult:
        definition = self.definitions.get(artifact_type)
        if not definition:
            return ValidationResult(False, (f"unknown artifact type {artifact_type}",))
        required = definition.get("required_fields", [])
        errors: list[str] = []
        try:
            rows = read_jsonl(path)
        except Exception as exc:
            return ValidationResult(False, (f"E_ARTIFACT_INVALID: {exc}",))
        if not rows:
            errors.append("E_ARTIFACT_INVALID: JSONL is empty")
        for index, row in enumerate(rows, 1):
            missing = [field for field in required if field not in row]
            if missing:
                errors.append(f"E_SCHEMA_MISMATCH: line {index} missing {','.join(missing)}")
        if artifact_type == "candidates":
            if len(rows) != 4:
                errors.append(f"E_SCHEMA_MISMATCH: candidates requires exactly 4 rows, got {len(rows)}")
            candidate_ids = [row.get("candidate_id") for row in rows]
            if len(set(candidate_ids)) != len(candidate_ids):
                errors.append("E_SCHEMA_MISMATCH: candidate_id values must be unique")
            worker_ids = [row.get("worker_id") for row in rows]
            if set(worker_ids) != {"worker-00", "worker-01", "worker-02", "worker-03"}:
                errors.append("E_SCHEMA_MISMATCH: candidates must cover worker-00 through worker-03 exactly once")
            errors.extend(validate_candidate_architecture(rows))
        if artifact_type == "literature":
            allowed_workers = {"worker-00", "worker-01", "worker-02", "worker-03"}
            for index, row in enumerate(rows, 1):
                workers = row.get("worker_ids", [row.get("worker_id")])
                if not isinstance(workers, list) or not workers or not set(workers).issubset(allowed_workers):
                    errors.append(f"E_SCHEMA_MISMATCH: literature line {index} lacks valid worker provenance")
        return ValidationResult(not errors, tuple(errors))


def control_envelope(artifact_type: str, run_id: str, producer: str, inputs: list[dict[str, str]], **fields: Any) -> dict[str, Any]:
    from kvresearch.util import utcnow

    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "run_id": run_id,
        "created_at": utcnow(),
        "producer": producer,
        "inputs": inputs,
        "status": fields.pop("status", "complete"),
        **fields,
    }
