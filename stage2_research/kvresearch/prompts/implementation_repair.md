Repair the already selected candidate in place; do not create a new research
cycle and do not change its scientific mechanism.  Read the validation artifact
and every per-task stdout/stderr path under `checks.model_smoke.tasks`.  These
are authoritative failures from a real, non-scoring POD invocation using one
locked row per planned task.

Automatically fix all implementation-readiness defects exposed by the smoke
gate, including missing candidate artifacts/codebooks, wrong paths, module or
ABI resolution, extension builds, CLI/batch/sample-count assumptions, dataset
routing, candidate/control dispatch, telemetry fields, storage accounting,
identity coverage, and atomic output publication.  Produce every required
artifact inside the isolated workspace; never modify either VecInfer baseline.
Do not merely catch, rename, suppress, or classify an exception, and do not
claim that a later 32-row evaluation will repair it.

After each bounded repair:

1. run `python3 -m kvresearch snapshot-implementation` for this same cycle and
   selected candidate;
2. run `python3 -m kvresearch validate-cycle` for this same cycle;
3. inspect the newly content-addressed real-POD smoke result;
4. continue repairing until `validation.all_hard_checks_pass=true` and every
   planned smoke task has exactly one complete row with identity, token counts,
   actual KV bytes, and exit code zero.

The smoke set is non-scoring and may use its bounded math liveness cap.  Passing
it provides no accuracy or bit conclusion; it only authorizes the fixed
small-sample evaluation.  Runtime/environment transients may be retried after a
fresh check.  A mechanism that genuinely requires an unavailable scientific
input (for example an unauthorized calibration corpus) may not fabricate that
input or silently become another mechanism: record concrete evidence and fail
closed so selection can choose an executable candidate in the same cycle.

When `diagnostic_tests.selector_rule` is declared, repair until every task
reports positive applicable and changed rollover coverage and a real bounded
independent reference result under
`selector_reference_oracle={"independent_fp32": true, "checks": N,
"mismatches": 0, "agreement": true}` with `N > 0`.  A constant agreement
boolean is not evidence and is rejected by the host before scored execution.

Finish by leaving the current implementation manifest at the requested output
path.  It must describe the final repaired workspace snapshot that passed the
real-POD smoke gate.
