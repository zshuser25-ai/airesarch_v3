from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from kvresearch.util import path_ref, read_json, read_jsonl, sha256_file


_NON_PROMOTABLE_MARKERS = (
    "non-promotable",
    "non promotable",
    "cannot itself be promoted",
    "cannot be promoted",
    "cannot open formal",
)


def diagnostic_contract(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the machine-enforced diagnostic semantics declared by a plan.

    Older generated plans placed ``non-promotable`` in prose rather than in a
    dedicated boolean.  Treat that declaration as binding as well; otherwise
    a diagnostic probe could accidentally become a promotion candidate merely
    because the producer omitted one optional field.
    """
    outputs = plan.get("outputs") if isinstance(plan.get("outputs"), list) else []
    output_requires_diagnostics = any(
        "diagnostic" in str(item.get("kind", "")).lower()
        for item in outputs if isinstance(item, dict)
    )
    tests = plan.get("diagnostic_tests")
    tests_required = isinstance(tests, dict) and bool(tests)
    explicit_non_promotable = plan.get("non_promotable") is True
    searchable = json.dumps(
        {
            "decision_rules": plan.get("decision_rules"),
            "diagnostic_tests": tests,
            "objective": plan.get("objective"),
            "evaluation_tier_rationale": plan.get("evaluation_tier_rationale"),
        },
        ensure_ascii=False,
    ).lower()
    prose_non_promotable = any(marker in searchable for marker in _NON_PROMOTABLE_MARKERS)
    non_promotable = explicit_non_promotable or prose_non_promotable
    return {
        "required": tests_required or output_requires_diagnostics or non_promotable,
        "non_promotable": non_promotable,
        "declared_tests": sorted(tests) if isinstance(tests, dict) else [],
        "reasons": [
            reason for condition, reason in (
                (tests_required, "diagnostic_tests"),
                (output_requires_diagnostics, "diagnostic_output"),
                (explicit_non_promotable, "non_promotable_flag"),
                (prose_non_promotable, "non_promotable_declaration"),
            ) if condition
        ],
    }


def _objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _numbers(child)


def _comparison_count(value: Any) -> int:
    count = 0
    for item in _objects(value):
        candidate = item.get("candidate")
        references = [item.get(name) for name in ("control", "baseline", "reference")]
        if (
            isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
            and any(
                isinstance(reference, (int, float)) and not isinstance(reference, bool)
                and math.isfinite(float(reference))
                for reference in references
            )
        ):
            count += 1
    return count


def _covered_tests(value: Any, declared: set[str]) -> set[str]:
    covered: set[str] = set()
    for item in _objects(value):
        tests = item.get("tests")
        if isinstance(tests, dict):
            covered.update(str(name) for name in tests if str(name) in declared)
        name = item.get("test")
        if isinstance(name, str) and name in declared:
            covered.add(name)
    return covered


def _selector_semantics(value: Any) -> dict[str, Any]:
    """Extract the fail-closed contract for a learned/rule-based selector.

    A top-level ``agreement: true`` assertion is not evidence.  The runtime
    must expose an explicitly independent FP32 oracle with a positive checked
    sample count and zero mismatches, in addition to proving that the real
    candidate path was both applicable and changed at least one assignment.
    """
    applicable: list[int] = []
    changed: list[int] = []
    oracle_records: list[dict[str, Any]] = []
    for item in _objects(value):
        for name, target in (
            ("applicable_rollovers", applicable),
            ("changed_rollovers", changed),
        ):
            number = item.get(name)
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                target.append(number)
        oracle = item.get("selector_reference_oracle")
        if isinstance(oracle, dict):
            oracle_records.append(oracle)

    oracle_valid = False
    oracle_checks = 0
    oracle_mismatches = 0
    for oracle in oracle_records:
        checks = oracle.get("checks")
        mismatches = oracle.get("mismatches")
        if (
            oracle.get("independent_fp32") is True
            and oracle.get("agreement") is True
            and isinstance(checks, int) and not isinstance(checks, bool) and checks > 0
            and isinstance(mismatches, int) and not isinstance(mismatches, bool)
            and mismatches == 0
        ):
            oracle_valid = True
            oracle_checks = max(oracle_checks, checks)
            oracle_mismatches = mismatches
    return {
        "applicable_rollovers": max(applicable, default=0),
        "changed_rollovers": max(changed, default=0),
        "oracle_records": len(oracle_records),
        "oracle_checks": oracle_checks,
        "oracle_mismatches": oracle_mismatches,
        "oracle_valid": oracle_valid,
    }


def _diagnostic_paths(attempt_dir: Path, task: str, *, remaining: bool = False) -> list[Path]:
    root = attempt_dir / ("diagnostics_remaining" if remaining else "diagnostics")
    paths = [root / f"{task}.json", root / f"{task}.jsonl"]
    task_dir = root / task
    if task_dir.is_dir():
        paths.extend(sorted(path for path in task_dir.rglob("*") if path.suffix in {".json", ".jsonl"}))
    return [path for path in paths if path.is_file()]


def _load_diagnostic(path: Path) -> Any:
    return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)


def _validated_input_refs(value: Any, attempt_dir: Path, task: str) -> tuple[list[dict[str, str]], list[str]]:
    valid: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    attempt_resolved = attempt_dir.resolve()
    for item in _objects(value):
        if not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            continue
        source = Path(item["path"])
        if not source.is_absolute():
            source = attempt_dir / source
        try:
            resolved = source.resolve()
            resolved.relative_to(attempt_resolved)
        except (OSError, ValueError):
            errors.append(f"diagnostic input escapes attempt: {source}")
            continue
        if not resolved.is_file():
            errors.append(f"diagnostic input missing: {resolved}")
            continue
        if task not in resolved.parts and resolved.name != f"{task}.jsonl":
            errors.append(f"diagnostic input is not task-bound for {task}: {resolved}")
            continue
        if sha256_file(resolved) != item["sha256"]:
            errors.append(f"diagnostic input hash drift for {task}: {resolved}")
            continue
        valid[str(resolved)] = path_ref(resolved)
    return list(valid.values()), errors


def validate_diagnostic_outputs(
    plan: dict[str, Any], attempt_dir: Path, *, remaining: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation for causal/observer outputs.

    A valid task summary must be explicitly complete, name its task, bind at
    least two immutable attempt inputs (the compared arms), contain no
    non-finite numeric value, and expose at least one finite candidate/control
    comparison.  Runtime role ledgers under ``mechanism/`` intentionally do
    not satisfy this contract.
    """
    contract = diagnostic_contract(plan)
    result: dict[str, Any] = {
        **contract,
        "attempt": str(attempt_dir),
        "phase": "remaining_70pct" if remaining else "initial_30pct",
        "valid": True,
        "tasks": {},
        "errors": [],
        "evidence_paths": [],
    }
    if not contract["required"]:
        return result

    for task in plan.get("datasets", []):
        paths = _diagnostic_paths(attempt_dir, str(task), remaining=remaining)
        task_errors: list[str] = []
        values: list[Any] = []
        for path in paths:
            try:
                values.append(_load_diagnostic(path))
            except Exception as exc:
                task_errors.append(f"invalid diagnostic file {path}: {exc}")
        combined: Any = values
        records = [item for value in values for item in _objects(value)]
        task_named = any(item.get("task") == task for item in records)
        # Nested scientific checkpoints legitimately use domain statuses such
        # as ``observed`` and ``not_applicable``.  They are not artifact
        # lifecycle markers.  Completion is therefore fail-closed on explicit
        # boolean ``complete`` fields only; treating every nested ``status`` as
        # an artifact status would reject otherwise valid observer output.
        completion_markers = [
            item.get("complete") for item in records if "complete" in item
        ]
        complete = bool(completion_markers) and all(marker is True for marker in completion_markers)
        numeric_values = list(_numbers(combined))
        finite = bool(numeric_values) and all(math.isfinite(float(value)) for value in numeric_values)
        comparisons = _comparison_count(combined)
        declared_tests = set(contract["declared_tests"])
        covered_tests = _covered_tests(combined, declared_tests)
        selector = _selector_semantics(combined) if "selector_rule" in declared_tests else None
        refs, ref_errors = _validated_input_refs(combined, attempt_dir, str(task))
        task_errors.extend(ref_errors)
        if not paths:
            task_errors.append(f"missing diagnostics for {task}")
        if paths and not task_named:
            task_errors.append(f"diagnostics do not name task {task}")
        if paths and not complete:
            task_errors.append(f"diagnostics are not complete for {task}")
        if paths and not finite:
            task_errors.append(f"diagnostics lack exclusively finite numeric evidence for {task}")
        if paths and comparisons == 0:
            task_errors.append(f"diagnostics lack a finite candidate/control comparison for {task}")
        if paths and declared_tests - covered_tests:
            task_errors.append(
                f"diagnostics do not cover declared tests for {task}: "
                + ",".join(sorted(declared_tests - covered_tests))
            )
        if paths and len(refs) < 2:
            task_errors.append(f"diagnostics require two hash-verified compared inputs for {task}")
        if paths and selector is not None:
            if selector["applicable_rollovers"] <= 0:
                task_errors.append(f"selector diagnostics have zero applicable rollover coverage for {task}")
            if selector["changed_rollovers"] <= 0:
                task_errors.append(f"selector diagnostics have zero changed rollover coverage for {task}")
            if not selector["oracle_valid"]:
                task_errors.append(
                    f"selector diagnostics lack an independent FP32 oracle with checks>0, "
                    f"mismatches=0, and agreement=true for {task}"
                )
        task_valid = not task_errors
        result["tasks"][str(task)] = {
            "valid": task_valid,
            "paths": [path_ref(path) for path in paths],
            "input_refs": refs,
            "finite_numeric_values": len(numeric_values),
            "comparison_count": comparisons,
            "covered_tests": sorted(covered_tests),
            "selector_semantics": selector,
            "errors": task_errors,
        }
        result["evidence_paths"].extend(str(path) for path in paths)
        result["errors"].extend(task_errors)
    result["valid"] = not result["errors"] and set(result["tasks"]) == set(plan.get("datasets", []))
    return result
