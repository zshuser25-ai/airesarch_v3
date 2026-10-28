from __future__ import annotations

from typing import Any


# These lanes are owned by the host.  Agent-authored family names are retained
# only as prose and never participate in quotas, selection, or cooldown.
ARCHITECTURE_LANES: dict[str, tuple[str, str, str]] = {
    "worker-00": ("architecture", "kv_bit_allocation", "kv_bit_allocation_architecture"),
    "worker-01": ("architecture", "residual_token_retention", "residual_token_retention_architecture"),
    "worker-02": ("architecture", "representation_decoder", "representation_decoder_architecture"),
    "worker-03": ("stock_control", "stock_vecinfer_control", "stock_vecinfer_control"),
}

SELECTABLE_FAMILIES = {
    "kv_bit_allocation_architecture",
    "residual_token_retention_architecture",
    "representation_decoder_architecture",
}


def candidate_contract(row: dict[str, Any]) -> tuple[str, str, str] | None:
    return ARCHITECTURE_LANES.get(str(row.get("worker_id", "")))


def canonical_mechanism_family(row: dict[str, Any]) -> str | None:
    """Return a host-owned broad family; never trust a renamed family label."""
    axis = str(row.get("architecture_axis", "")).strip().lower()
    by_axis = {value[1]: value[2] for value in ARCHITECTURE_LANES.values()}
    if axis in by_axis:
        return by_axis[axis]

    # Legacy artifacts predate architecture_axis. Collapse aliases for the
    # same K-codebook/assignment mechanism into one host-owned family.
    text = " ".join(
        str(row.get(key, "")) for key in (
            "mechanism_family", "mechanism", "candidate_id", "causal_chain",
        )
    ).lower()
    k_assignment_markers = (
        "centroid", "codebook assignment", "codebook_assignment", "radial",
        "cartesian", "query-aware k", "query_aware_k", "historical k",
        "historical_k", "k assignment", "k_assignment", "pq assignment",
    )
    if any(marker in text for marker in k_assignment_markers):
        return "k_only_codebook_assignment"
    if "residual" in text or "sink token" in text or "token retention" in text:
        return "residual_token_retention_architecture"
    if "bit allocation" in text or "k/v allocation" in text or "key/value allocation" in text:
        return "kv_bit_allocation_architecture"
    if any(marker in text for marker in ("decoder", "binary representation", "ternary representation")):
        return "representation_decoder_architecture"
    if "stock vecinfer" in text or "unmodified vecinfer" in text:
        return "stock_vecinfer_control"
    return None


def validate_candidate_architecture(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 4:
        return [f"E_ARCHITECTURE_QUOTA: expected exactly 4 lanes, got {len(rows)}"]
    by_worker = {str(row.get("worker_id", "")): row for row in rows}
    if set(by_worker) != set(ARCHITECTURE_LANES):
        return ["E_ARCHITECTURE_QUOTA: candidates must cover the four host-owned lanes exactly once"]
    for worker_id, expected in ARCHITECTURE_LANES.items():
        expected_kind, expected_axis, expected_family = expected
        row = by_worker[worker_id]
        if row.get("candidate_kind") != expected_kind:
            errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} candidate_kind must be {expected_kind}")
        if row.get("architecture_axis") != expected_axis:
            errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} architecture_axis must be {expected_axis}")
        if row.get("canonical_mechanism_family") != expected_family:
            errors.append(
                f"E_FAMILY_CLASSIFICATION: {worker_id} canonical_mechanism_family must be {expected_family}"
            )
        if canonical_mechanism_family(row) != expected_family:
            errors.append(f"E_FAMILY_CLASSIFICATION: {worker_id} does not resolve to its host family")
        if expected_kind == "architecture":
            change = row.get("architecture_change")
            if not isinstance(change, dict):
                errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} requires architecture_change")
            else:
                if change.get("parameter_only") is not False:
                    errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} parameter_only must be false")
                if not isinstance(change.get("state_before"), str) or not change["state_before"].strip():
                    errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} requires state_before")
                if not isinstance(change.get("state_after"), str) or not change["state_after"].strip():
                    errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} requires state_after")
                components = change.get("changed_runtime_components")
                if not isinstance(components, list) or not components or not all(
                    isinstance(item, str) and item.strip() for item in components
                ):
                    errors.append(f"E_ARCHITECTURE_QUOTA: {worker_id} requires changed_runtime_components")
        else:
            invariants = row.get("control_invariants")
            if not isinstance(invariants, dict) or not (
                invariants.get("selectable") is False
                and invariants.get("codebook_source") == "baseline.centroids"
                and invariants.get("assignment_policy") == "squared_euclidean"
                and invariants.get("candidate_artifacts_forbidden") is True
            ):
                errors.append(f"E_STOCK_CONTROL_INVALID: {worker_id} stock control invariants are incomplete")
    return errors
