from __future__ import annotations

from typing import Any

from kvresearch.config import Config


def vecinfer_reference_bytes(
    config: Config,
    total_tokens: int,
    sink_tokens: int,
    *,
    prefill_tokens: int | None = None,
    output_tokens: int | None = None,
) -> int:
    values = config.section("storage")["vecinfer"]
    layers = int(values["layers"])
    heads = int(values["kv_heads"])
    dimension = int(values["head_dimension"])
    scalar_bytes = int(values["scalar_bytes"])
    residual_capacity = int(values["residual_tokens"])
    chunk = residual_capacity - sink_tokens
    if prefill_tokens is not None and output_tokens is not None and chunk > 0:
        # VecInfer prefill encodes complete non-sink chunks of `chunk` tokens.
        # During decode it flushes the full residual arena (including the sink
        # rows) only *before* accepting another token.  The final generated
        # token is returned by the LM head and is not fed back into the cache.
        # Reproduce that allocation lifecycle exactly instead of estimating it
        # from total length; otherwise boundary requests are off by 16 or 112
        # token-code rows under the LongBench sink=16 protocol.
        prefill_remaining = max(0, int(prefill_tokens) - sink_tokens)
        prefill_chunks, residual_non_sink = divmod(prefill_remaining, chunk)
        fed_decode_tokens = max(0, int(output_tokens) - 1)
        decode_flushes = (
            (residual_non_sink + fed_decode_tokens - 1) // chunk
            if fed_decode_tokens > 0
            else 0
        )
        quantized_tokens = prefill_chunks * chunk + decode_flushes * residual_capacity
    else:
        remaining = max(0, total_tokens - sink_tokens)
        quantized_tokens = (remaining // chunk) * chunk if chunk > 0 else 0
    key_codes = heads * quantized_tokens * int(values["key_subspaces"])
    value_codes = heads * quantized_tokens * int(values["value_subspaces"])
    residual = 2 * heads * residual_capacity * dimension * scalar_bytes
    return layers * (key_codes + value_codes + residual)


def normalize_storage_record(config: Config, task: str, row: dict[str, Any], request_id: str, theoretical_bits: float) -> dict[str, Any]:
    storage = config.section("storage")["vecinfer"]
    total_tokens = int(row.get("total_tokens") or (int(row.get("prefill_tokens", 0)) + int(row.get("output_tokens", 0))))
    actual = row.get("actual_pq_kv_cache_bytes")
    actual_bytes = int(actual) if isinstance(actual, (int, float)) else None
    sink = int(storage["math_sink_tokens"] if task in {"gsm8k", "math500"} else storage["longbench_sink_tokens"])
    prefill = row.get("prefill_tokens")
    output = row.get("output_tokens")
    reference = vecinfer_reference_bytes(
        config,
        total_tokens,
        sink,
        prefill_tokens=int(prefill) if isinstance(prefill, (int, float)) else None,
        output_tokens=int(output) if isinstance(output, (int, float)) else None,
    )
    return {
        "request_id": request_id,
        "sequence_length": total_tokens,
        "layers": int(storage["layers"]),
        "kv_elements": 2 * int(storage["layers"]) * int(storage["kv_heads"]) * total_tokens * int(storage["head_dimension"]),
        "payload_bytes": None,
        "index_bytes": None,
        "scale_bytes": None,
        "zero_point_bytes": None,
        "metadata_bytes": None,
        "residual_bytes": None,
        "padding_bytes": None,
        "alignment_bytes": None,
        "codebook_bytes": None,
        "unattributed_bytes": actual_bytes,
        "logical_total_bytes": actual_bytes,
        "allocated_total_bytes": actual_bytes,
        "theoretical_average_bits": theoretical_bits,
        "vecinfer_reference_bytes": reference,
        "shared_cost_policy": "VecInfer raw record exposes total request cache bytes; component split is unavailable and retained as unattributed_bytes",
        "actual_constraint_pass": actual_bytes is not None and actual_bytes <= reference,
    }
