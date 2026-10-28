Implement only the selected candidate in an isolated workspace. Never modify baseline directories.

Implement the control arm as a genuinely separate stock VecInfer path. It must load the host-provided `VECINFER_STOCK_CODEBOOK_ROOT`, verify `VECINFER_STOCK_CODEBOOK_MANIFEST_SHA256`, use `VECINFER_STOCK_ASSIGNMENT_POLICY=squared_euclidean`, and never install a candidate loader or candidate codebook root. Every published control row must copy those verified values into `arm_role="stock_vecinfer_control"`, `stock_vecinfer_codebook_root`, `stock_vecinfer_codebook_manifest_sha256`, and `stock_vecinfer_assignment_policy`. Merely renaming a candidate arm to `unmodified_vecinfer` is a hard technical failure.

When the inputs include a prior development `implementation_manifest.json`, its
passing `validation.json`, and a `PROMOTE` decision, this is a formal
carry-forward. Reuse the exact promoted source tree and artifacts in the new
cycle workspace; do not reimplement, tune, simplify, or otherwise change the
algorithm. Verify every prior changed-file hash and source-tree hash before
copying, then snapshot the unchanged carried implementation against the new
formal plan. Any hash mismatch is a technical failure to repair, never license
to substitute a different candidate.

Implementation is not complete after syntax checks or a mocked unit test. Before handing the candidate to the fixed small-sample evaluator, it must pass the host's real POD implementation-smoke gate: one locked, non-scoring row for every task in the plan through the exact `scripts.modeldb.main_pq` adapter path, including model load, all required codebooks/artifacts, candidate dispatch, prediction fields, actual-KV-byte telemetry, and atomic publication. Run `snapshot-implementation`, then `validate-cycle`; if `checks.model_smoke` fails, read every referenced stdout/stderr log, repair the same candidate in place, resnapshot, and rerun validation until every smoke task passes. Do not defer a repair to the 32-row evaluation and do not treat a smoke failure as an accuracy/bit result.

Only when the current selected candidate explicitly requires fitting query-aware K centroids from calibration data, use the authorized read-only cached `mit-han-lab/pile-val-backup` validation corpus and the baseline recipe, hash the selected blocks, prove exact disjointness, and fit all 32 codebooks. For later candidates that cite already locked cycle artifacts, verify and consume those exact path/SHA inputs and implement only the transformation named by the current plan; do not repeat an archived calibration phase or substitute its mechanism. Every required K/V artifact must still exist, be finite, have the declared shape/dtype and hash, and be loaded by smoke rather than represented by a stub.

Confirm execution closure against `kvresearch.vecinfer.VecInferAdapter.commands()` before snapshotting. The host launches one ordinary `scripts.modeldb.main_pq` process per task; it does not run arbitrary extra phases merely because they appear in the experiment plan. Do not snapshot an implementation that requires a missing cross-task calibration, identity, diagnostic, reducer, protected-map, or paired-control artifact. Either implement the prerequisite within the authorized workspace and actual invocation contract with deterministic race-free artifacts, or use an already locked artifact whose path and hash exist. If closure fails, repair this same candidate in this same cycle and rerun snapshot plus smoke; do not advance to another research candidate on a technical failure.

For a diagnostic plan, implement the observer itself, not merely the replay arms. Before a task process returns success it must publish `attempts/{attempt}/diagnostics/{task}.json` (or `.jsonl`) with `task`, `complete: true`, named coverage of every `diagnostic_tests` key, finite observer statistics, one or more objects such as `{"metric": "qk_logit_l2", "candidate": 0.7, "control": 0.2}`, and two hash-verifiable `{path, sha256}` bindings to the compared candidate and control inputs inside that attempt. The host deliberately rejects `mechanism/*.jsonl` execution metadata as causal evidence.

If the plan declares `diagnostic_tests.selector_rule`, a boolean assertion is not an oracle. Every per-task diagnostic must report positive `applicable_rollovers` and `changed_rollovers`, plus `selector_reference_oracle={"independent_fp32": true, "checks": <positive integer>, "mismatches": 0, "agreement": true}` produced by a separately implemented bounded FP32 reference comparison. The host rejects zero coverage, a missing oracle, or a hard-coded agreement flag as technical invalidity before scored evaluation.

For paired execution, require equal fixed input identity, prompt, model, seed, decoding configuration and maximum-generation budget, but do not require equal prediction text, EOS position, output-token count, total-token count or final logical cache length. Those are measured algorithm outcomes. Compare actual KV storage only at a matched logical length, using a common post-prefill checkpoint or deterministic fixed-length storage probe/normalization, not unequal natural end-of-generation totals. Charge all algorithm-required persistent tensors, metadata/routing state and bounded workspace; exclude host-side logs/JSON and observer-only bookkeeping. Preserve process-wide allocated/reserved peaks as engineering telemetry, but do not fail the bit gate on allocator noise unless the measurement isolates candidate-owned KV deltas at the matched checkpoint.

For every candidate, preserve the mechanism selected in the current cycle and honor the plan's shared `VECINFER_GENERATION_MAX_LENGTH=2048` in both math quality arms. Validate the bound before model load, emit the configured maximum and cap finish reason per row, and keep a regression fixture proving both arms share the bound while the 16K matched-storage probe remains unchanged. Do not silently substitute an archived mechanism from another cycle, and do not change the fixed identities, scorer, seeds, sampling parameters, stop IDs, or storage layout.

The paired supervisor must always support the active plan namespace: development/initial output under `raw` with sibling `joins`. When and only when `evaluation_tier="formal"`, it must additionally support the conditional completion namespace `raw_remaining` with sibling `joins_remaining`, deriving every arm, build, mechanism, storage and probe path from the selected namespace so 70% cannot overwrite or reuse 30%. Do not add unrelated formal behavior to a development candidate.

Only for a formal carry-forward plan, remove development-only row-count and batch assumptions without changing the promoted algorithm: derive ordered identities from the active locked manifest, accept the executor's `[16,8,4,2,1]` OOM backoff, use the same selected batch in both quality arms, retain batch-1 storage probes, and make eight task parents race-free. A development plan remains fixed at five tasks × 32 rows and batch 1; it must not claim or implement formal 30%/70% work early.

If a formal plan inherits a workspace with development-only guards in `scripts/modeldb/main_pq.py`, generalize the fixed-row and batch checks and prove real batch execution with POD tests before formal scoring. This paragraph is inactive for development plans, where the locked workload is exactly 32 identities per task and batch 1.

If the cycle workspace does not exist, first run `python3 -m kvresearch prepare-workspace --run <run_dir> --cycle <cycle>`. If it already exists, repair it in place rather than recreating it. Modify only that cycle's workspace. Preserve VecInfer's CLI and evaluator interfaces. Every raw prediction row must retain `_fixed_sample`, token counts, timing, peak memory, and `actual_pq_kv_cache_bytes`; the byte value must include every candidate-owned KV allocation and may not be estimated from the configured bit label.

During implementation, never create or write any numbered
`cycles/<cycle>/attempts/<NNN>` directory; those namespaces are exclusively
reserved for scored executor attempts after validation passes, including fresh
technical-repair attempts beyond the research-attempt budget. Put any additional manual batch or
integration probes under `cycles/<cycle>/implementation_tests/<content-hash>`
or rely on the host-managed `implementation_smoke/smoke-*` namespace. A
numbered directory containing build, raw, arm, diagnostic, join, or log state
before preflight is technical contamination and must not be reused.

Run cheap syntax/import checks using the locked POD runtime. If a CUDA extension is changed, build it inside the workspace and set `execution.extension_path` in the plan to the resulting compatible extension directory. Then run `python3 -m kvresearch snapshot-implementation --run <run_dir> --cycle <cycle> --candidate-id <selected_id>`. That deterministic command must create the requested implementation manifest; do not hand-author hashes.
