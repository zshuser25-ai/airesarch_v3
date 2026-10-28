from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvresearch.config import Config
from kvresearch.mechanisms import canonical_mechanism_family
from kvresearch.schema import ArtifactValidator
from kvresearch.util import latest_numbered_dir, read_json, read_jsonl, sha256_file


@dataclass(frozen=True)
class Action:
    state: str
    step: str
    kind: str
    inputs: tuple[str, ...]
    output: str | None
    argv: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "step": self.step,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "output": self.output,
            "argv": list(self.argv),
            "notes": self.notes,
        }


AGENT_STEPS = {
    "literature": "evidence_literature.md",
    "evidence_synthesis": "evidence_synthesis.md",
    "problem_diagnosis": "problem_diagnosis.md",
    "candidates": "candidates.md",
    "selection": "selection.md",
    "experiment_plan": "experiment_plan.md",
    "implementation_manifest": "implementation.md",
    "implementation_repair": "implementation_repair.md",
    "execution_repair": "execution_repair.md",
}


def _artifact_binds_current_inputs(artifact: dict[str, Any], paths: tuple[Path, ...]) -> bool:
    references = {
        str(Path(item.get("path", "")).resolve()): item.get("sha256")
        for item in artifact.get("inputs", [])
        if isinstance(item, dict) and item.get("path")
    }
    # Older imported artifacts and small unit-test fixtures may not carry
    # provenance refs. When a ref is present, however, it is binding and must
    # never silently authorize a changed plan or implementation snapshot.
    for path in paths:
        expected = references.get(str(path.resolve()))
        if expected is not None and (not path.is_file() or expected != sha256_file(path)):
            return False
    return True


def _cycle_budget_reached(configured_max_cycles: int, current_cycle: int) -> bool:
    """Return whether the operational cycle ceiling has been reached.

    The scientific config remains immutable inside a run.  Goal-mode may lift
    that operational ceiling without rewriting the locked protocol by setting
    ``KVRESEARCH_MAX_CYCLES_OVERRIDE=unbounded``.  Numeric overrides retain the
    previous extend-only behavior.
    """
    override = os.environ.get("KVRESEARCH_MAX_CYCLES_OVERRIDE")
    if override:
        normalized = override.strip().lower()
        if normalized in {"unbounded", "unlimited", "infinite", "inf"}:
            return False
        max_cycles = max(configured_max_cycles, int(override))
    else:
        max_cycles = configured_max_cycles
    return current_cycle + 1 >= max_cycles


def _consume_remote_resume() -> bool:
    """Consume one explicit remote retry authorization in this auto process."""
    value = os.environ.pop("KVRESEARCH_RESUME_REMOTE_ONCE", "")
    return value.strip().lower() in {"1", "true", "yes"}


class Navigator:
    def __init__(self, config: Config):
        self.config = config
        self.validator = ArtifactValidator(config.catalog_path)

    def _command(self, run_dir: Path, step: str, *extra: str, output: Path | None = None, inputs: tuple[Path, ...] = ()) -> Action:
        return Action(
            state="READY",
            step=step,
            kind="command",
            inputs=tuple(str(path) for path in inputs),
            output=str(output) if output else None,
            argv=("kvresearch", "--config", str(self.config.path), step, "--run", str(run_dir), *extra),
        )

    def _agent(self, step: str, inputs: tuple[Path, ...], output: Path) -> Action:
        return Action(
            state="READY",
            step=step,
            kind="agent",
            inputs=tuple(str(path) for path in inputs),
            output=str(output),
            notes=f"prompt={Path(__file__).resolve().parent / 'prompts' / AGENT_STEPS[step]}",
        )

    def _valid_control(self, path: Path, artifact_type: str) -> tuple[bool, str]:
        if not path.is_file():
            return False, "missing"
        result = self.validator.validate_control(path, artifact_type)
        return result.valid, "; ".join(result.errors)

    def _valid_jsonl(self, path: Path, artifact_type: str) -> tuple[bool, str]:
        if not path.is_file():
            return False, "missing"
        result = self.validator.validate_jsonl(path, artifact_type)
        return result.valid, "; ".join(result.errors)

    @staticmethod
    def _probe_references_are_current(path: Path) -> bool:
        """Reject probe records whose structured file bindings are stale.

        A run directory can be resumed after a cycle number has been rebuilt.
        Host-authored ``evidence/cycle*.json`` from the earlier incarnation may
        then still exist.  Passing such a probe to a diagnostician is unsafe:
        an old correction can incorrectly supersede a newer, complete result.
        Require every structured absolute path reference to exist and every
        adjacent SHA-256 binding to match before exposing a probe to an agent.
        """
        try:
            data = read_json(path)
        except (OSError, ValueError):
            return False

        path_keys = {"path", "control", "candidate", "baseline", "input", "output", "partial_path"}

        def walk(value: Any) -> bool:
            if isinstance(value, dict):
                bound_path = value.get("path")
                bound_sha = value.get("sha256")
                if isinstance(bound_path, str) and bound_path.startswith("/"):
                    resolved = Path(bound_path)
                    if not resolved.is_file():
                        return False
                    if isinstance(bound_sha, str) and sha256_file(resolved) != bound_sha:
                        return False
                for key, item in value.items():
                    if isinstance(item, str) and item.startswith("/") and (
                        key in path_keys or key.endswith("_path")
                    ):
                        if not Path(item).exists():
                            return False
                    if not walk(item):
                        return False
                return True
            if isinstance(value, list):
                return all(walk(item) for item in value)
            return True

        return walk(data)

    @classmethod
    def _diagnosis_evidence(cls, run_dir: Path) -> tuple[Path, ...]:
        """Return deterministic, host-authored evidence that diagnoses later cycles.

        These records are intentionally separate from the one-time literature
        synthesis: they capture runtime probes and repaired measurement semantics
        discovered during the loop.  Passing them as formal agent inputs both
        exposes the newest facts and changes the task fingerprint when evidence
        is added, so stale worker results cannot be reused.
        """
        evidence_dir = run_dir / "evidence"
        return tuple(
            path for path in sorted(evidence_dir.glob("cycle*.json"))
            if cls._probe_references_are_current(path)
        )

    def _problem_diagnosis_inputs(self, run_dir: Path, source_cycle: Path | None) -> tuple[Path, ...]:
        """Build one restart-stable evidence set for a new cycle diagnosis.

        ``next()`` handles recovery after an interruption while
        ``_post_execution()`` handles the uninterrupted transition.  Keeping
        their input construction here prevents a restart from silently
        dropping the previous cycle's learned diagnostic evidence or changing
        the agent-task fingerprint.
        """
        inputs: tuple[Path, ...] = (
            run_dir / "request.md",
            run_dir / "evidence" / "synthesis.json",
            run_dir / "inputs.lock.json",
        )
        if source_cycle is not None:
            decision = source_cycle / "decision.json"
            valid, _ = self._valid_control(decision, "decision")
            if valid:
                inputs += (decision,)
            lessons = source_cycle / "lessons.jsonl"
            valid, _ = self._valid_jsonl(lessons, "lessons")
            if valid:
                inputs += (lessons,)
        return inputs + self._diagnosis_evidence(run_dir)

    def next(self, run_dir: Path) -> Action:
        run_dir = run_dir.resolve()
        terminal = run_dir / "terminal.json"
        if terminal.is_file():
            valid, why = self._valid_control(terminal, "terminal")
            return Action("DONE" if valid else "INVALID", "terminal", "terminal", (str(terminal),), None, notes=why)

        root_files = (("run_manifest", run_dir / "run_manifest.json"), ("inputs_lock", run_dir / "inputs.lock.json"))
        for artifact_type, path in root_files:
            valid, why = self._valid_control(path, artifact_type)
            if not valid:
                return Action("INVALID", artifact_type, "repair_input", (str(path),), str(path), notes=why)
        request = run_dir / "request.md"
        if not request.is_file() or not request.read_text(encoding="utf-8").strip():
            return Action("INVALID", "request", "repair_input", (), str(request), notes="missing original request")

        literature = run_dir / "evidence" / "literature.jsonl"
        valid, _ = self._valid_jsonl(literature, "literature")
        if not valid:
            return self._agent("literature", (request, run_dir / "inputs.lock.json"), literature)
        synthesis = run_dir / "evidence" / "synthesis.json"
        valid, _ = self._valid_control(synthesis, "evidence_synthesis")
        if not valid:
            return self._agent("evidence_synthesis", (request, literature, run_dir / "inputs.lock.json"), synthesis)

        cycles_root = run_dir / "cycles"
        cycle = latest_numbered_dir(cycles_root) or cycles_root / "000"
        previous_cycle: Path | None = None
        previous_decision: Path | None = None
        if int(cycle.name) > 0:
            previous_cycle = cycles_root / f"{int(cycle.name)-1:03d}"
            previous_decision = previous_cycle / "decision.json"

        # A valid development PROMOTE is a protocol transition, not a request
        # for another research hypothesis.  The next numbered cycle carries
        # the exact selected implementation into formal 30% evaluation.
        # Skipping diagnosis/candidate generation here prevents an already
        # promoted method from being silently replaced before formal testing.
        formal_promotion_source: Path | None = None
        if previous_cycle and previous_decision and previous_decision.is_file():
            source_plan = previous_cycle / "experiment_plan.json"
            source_decision_valid, _ = self._valid_control(previous_decision, "decision")
            source_plan_valid, _ = self._valid_control(source_plan, "experiment_plan")
            if (
                source_decision_valid
                and source_plan_valid
                and read_json(previous_decision).get("action") == "PROMOTE"
                and read_json(source_plan).get("evaluation_tier") == "development"
            ):
                formal_promotion_source = previous_cycle

        if formal_promotion_source is not None:
            selection = formal_promotion_source / "selection.json"
            plan = cycle / "experiment_plan.json"
            valid, _ = self._valid_control(plan, "experiment_plan")
            if not valid:
                return self._agent(
                    "experiment_plan",
                    (
                        selection,
                        run_dir / "inputs.lock.json",
                        formal_promotion_source / "decision.json",
                        formal_promotion_source / "experiment_plan.json",
                        formal_promotion_source / "implementation_manifest.json",
                        formal_promotion_source / "validation.json",
                    ),
                    plan,
                )
            implementation = cycle / "implementation_manifest.json"
            valid, _ = self._valid_control(implementation, "implementation_manifest")
            if not valid:
                return self._agent(
                    "implementation_manifest",
                    (
                        plan,
                        selection,
                        formal_promotion_source / "implementation_manifest.json",
                        formal_promotion_source / "validation.json",
                        formal_promotion_source / "decision.json",
                    ),
                    implementation,
                )
            return self._after_implementation(run_dir, cycle, plan, selection, implementation)

        problem = cycle / "problem_diagnosis.json"
        valid, _ = self._valid_control(problem, "problem_diagnosis")
        if not valid:
            inputs = self._problem_diagnosis_inputs(run_dir, previous_cycle)
            return self._agent("problem_diagnosis", inputs, problem)
        candidates = cycle / "candidates.jsonl"
        valid, _ = self._valid_jsonl(candidates, "candidates")
        cooldown = read_json(previous_decision).get("cooldown", {}) if previous_decision and previous_decision.is_file() else {}
        active = list(cooldown.get("active_families", []))
        if cooldown.get("triggered") and cooldown.get("mechanism_family"):
            active.append({"mechanism_family": cooldown["mechanism_family"], "until_cycle": cooldown.get("cooldown_until_cycle")})
        blocked_families = {str(item.get("mechanism_family", "")).strip().lower() for item in active
                            if int(item.get("until_cycle") or -1) >= int(cycle.name)}
        if not valid:
            extra = (previous_decision,) if previous_decision else ()
            return self._agent("candidates", (problem, synthesis) + extra, candidates)
        selection = cycle / "selection.json"
        valid, _ = self._valid_control(selection, "selection")
        if valid and blocked_families:
            selected_id = read_json(selection).get("selected_candidate_id")
            selected_row = next(
                (row for row in read_jsonl(candidates) if row.get("candidate_id") == selected_id),
                {},
            )
            valid = canonical_mechanism_family(selected_row) not in blocked_families
        if not valid:
            extra = (previous_decision,) if previous_decision else ()
            return self._agent("selection", (problem, candidates) + extra, selection)
        plan = cycle / "experiment_plan.json"
        valid, _ = self._valid_control(plan, "experiment_plan")
        if not valid:
            return self._agent("experiment_plan", (selection, run_dir / "inputs.lock.json"), plan)
        implementation = cycle / "implementation_manifest.json"
        valid, _ = self._valid_control(implementation, "implementation_manifest")
        if not valid:
            return self._agent("implementation_manifest", (plan, selection), implementation)
        return self._after_implementation(run_dir, cycle, plan, selection, implementation)

    def _after_implementation(
        self,
        run_dir: Path,
        cycle: Path,
        plan: Path,
        selection: Path,
        implementation: Path,
    ) -> Action:
        validation = cycle / "validation.json"
        valid, _ = self._valid_control(validation, "validation")
        if valid:
            valid = _artifact_binds_current_inputs(read_json(validation), (plan, implementation))
        if not valid:
            return self._command(run_dir, "validate-cycle", "--cycle", cycle.name, output=validation, inputs=(plan, implementation))
        validation_data = read_json(validation)
        if not validation_data.get("all_hard_checks_pass", False):
            return self._agent(
                "implementation_repair",
                (plan, selection, implementation, validation),
                implementation,
            )

        attempts = cycle / "attempts"
        attempt = latest_numbered_dir(attempts) or attempts / "000"
        # Numbered attempt directories are reserved for the executor.  Manual
        # implementation probes must never seed them.  If an interrupted or
        # out-of-contract probe left content before preflight, preserve that
        # evidence and advance to a fresh number instead of reusing build or
        # output state as a scored attempt.
        if (
            attempt.is_dir()
            and not (attempt / "preflight.json").is_file()
            and any(attempt.iterdir())
        ):
            next_index = int(attempt.name) + 1
            next_attempt = attempts / f"{next_index:03d}"
            return self._command(
                run_dir, "preflight", "--cycle", cycle.name, "--attempt", next_attempt.name,
                output=next_attempt / "preflight.json", inputs=(plan, implementation),
            )
        # A completed or partial attempt is immutable evidence.  Check its
        # bound implementation before validating the old preflight: a repair
        # commonly changes both the implementation and the plan, which makes
        # that preflight's input hash stale.  Re-running preflight in the same
        # numbered directory would otherwise overwrite the evidence we are
        # explicitly trying to preserve.
        attempt_manifest = attempt / "run_manifest.json"
        if attempt_manifest.is_file():
            manifest_data = read_json(attempt_manifest)
            executed_inputs = manifest_data.get("input_hashes") or {}
            executed_implementation = executed_inputs.get("implementation")
            current_implementation = sha256_file(implementation)
            executor_path = Path(__file__).resolve().with_name("executor.py")
            host_executor_changed = (
                executed_inputs.get("host_executor") is not None
                and executed_inputs.get("host_executor") != sha256_file(executor_path)
            )
            host_config_changed = (
                executed_inputs.get("host_config") is not None
                and executed_inputs.get("host_config") != sha256_file(self.config.path)
            )
            remote_path = Path(__file__).resolve().with_name("remote.py")
            host_remote_changed = (
                executed_inputs.get("host_remote") is not None
                and executed_inputs.get("host_remote") != sha256_file(remote_path)
            )
            if (
                executed_implementation != current_implementation
                or host_executor_changed
                or host_config_changed
                or host_remote_changed
            ):
                next_index = int(attempt.name) + 1
                # Technical repair attempts are immutable evidence but do not
                # consume a research-attempt budget.  Always allocate the next
                # monotonically numbered directory after a new implementation
                # snapshot, even when earlier technical attempts already
                # reached the configured research retry count.
                next_attempt = attempts / f"{next_index:03d}"
                return self._command(
                    run_dir, "preflight", "--cycle", cycle.name, "--attempt", next_attempt.name,
                    output=next_attempt / "preflight.json", inputs=(plan, implementation, attempt_manifest),
                )
        preflight = attempt / "preflight.json"
        valid, _ = self._valid_control(preflight, "preflight")
        if not valid:
            return self._command(run_dir, "preflight", "--cycle", cycle.name, "--attempt", attempt.name, output=preflight, inputs=(plan,))
        preflight_data = read_json(preflight)
        if not preflight_data.get("requested_resources_available"):
            if _consume_remote_resume():
                next_index = int(attempt.name) + 1
                next_attempt = attempts / f"{next_index:03d}"
                return self._command(
                    run_dir, "preflight", "--cycle", cycle.name,
                    "--attempt", next_attempt.name,
                    output=next_attempt / "preflight.json", inputs=(plan, preflight),
                )
            return Action(
                "WAIT", "remote_resources", "terminal", (str(preflight),), None,
                notes="remote resources are unavailable; preserve the attempt and retry only after an explicit resume",
            )
        exit_path = attempt / "exit.json"
        if not exit_path.is_file():
            if attempt_manifest.is_file():
                return self._command(run_dir, "monitor", "--cycle", cycle.name, "--attempt", attempt.name, output=exit_path, inputs=(attempt_manifest,))
            return self._command(run_dir, "execute", "--cycle", cycle.name, "--attempt", attempt.name, output=exit_path, inputs=(plan, preflight))
        exit_data = read_json(exit_path)
        if exit_data.get("status") == "failed" and exit_data.get("error_code") == "E_REMOTE_ENV":
            if _consume_remote_resume():
                next_index = int(attempt.name) + 1
                next_attempt = attempts / f"{next_index:03d}"
                return self._command(
                    run_dir, "preflight", "--cycle", cycle.name,
                    "--attempt", next_attempt.name,
                    output=next_attempt / "preflight.json", inputs=(plan, exit_path),
                )
            return Action(
                "WAIT", "remote_resources", "terminal", (str(exit_path),), None,
                notes="remote execution failed; retry only after resource and orphan-process audit",
            )
        if exit_data.get("status") == "failed" or int(exit_data.get("exit_code", 1)) != 0:
            return self._agent(
                "execution_repair",
                (plan, selection, implementation, validation, exit_path),
                implementation,
            )
        return self._post_execution(run_dir, cycle, plan, implementation, validation, exit_path)

    def _post_execution(self, run_dir: Path, cycle: Path, plan: Path, implementation: Path, validation: Path, exit_path: Path | None = None) -> Action:
        selection = cycle / "selection.json"
        if not selection.is_file() and int(cycle.name) > 0:
            prior = cycle.parent / f"{int(cycle.name)-1:03d}"
            prior_decision = prior / "decision.json"
            prior_plan = prior / "experiment_plan.json"
            if (
                prior_decision.is_file()
                and prior_plan.is_file()
                and read_json(prior_decision).get("action") == "PROMOTE"
                and read_json(prior_plan).get("evaluation_tier") == "development"
            ):
                selection = prior / "selection.json"
        metrics = cycle / "metrics.json"
        valid, _ = self._valid_control(metrics, "metrics")
        if not valid:
            inputs = (plan, validation) + ((exit_path,) if exit_path else ())
            return self._command(run_dir, "aggregate", "--cycle", cycle.name, output=metrics, inputs=inputs)
        metrics_data = read_json(metrics)
        storage = metrics_data.get("storage_accounting_summary") or {}
        technical_storage_failure = (
            (metrics_data.get("theoretical_storage_pass") is not True and int(storage.get("records", 0) or 0) == 0)
            or (metrics_data.get("actual_storage_pass") is not True and
                (int(storage.get("records", 0) or 0) == 0 or int(storage.get("missing_actual_bytes", 0) or 0) > 0))
        )
        if metrics_data.get("status") != "complete" or metrics_data.get("failures") or technical_storage_failure:
            repair_inputs = (plan, selection, implementation, validation, metrics) + ((exit_path,) if exit_path else ())
            return self._agent("execution_repair", repair_inputs, implementation)
        registry = cycle / "verified_registry.json"
        valid, _ = self._valid_control(registry, "verified_registry")
        if not valid:
            return self._command(run_dir, "verify", "--cycle", cycle.name, output=registry, inputs=(metrics,))
        diagnosis = cycle / "experiment_diagnosis.json"
        valid, _ = self._valid_control(diagnosis, "experiment_diagnosis")
        if not valid:
            return self._command(run_dir, "diagnose", "--cycle", cycle.name, output=diagnosis, inputs=(metrics, registry))
        if read_json(diagnosis).get("measurement_valid") is not True:
            repair_inputs = (plan, selection, implementation, validation, metrics, registry, diagnosis) + ((exit_path,) if exit_path else ())
            return self._agent("execution_repair", repair_inputs, implementation)
        decision = cycle / "decision.json"
        valid, _ = self._valid_control(decision, "decision")
        if not valid:
            return self._command(run_dir, "decide", "--cycle", cycle.name, output=decision, inputs=(metrics, registry, diagnosis))
        lessons = cycle / "lessons.jsonl"
        valid, _ = self._valid_jsonl(lessons, "lessons")
        if not valid:
            return self._command(run_dir, "learn", "--cycle", cycle.name, output=lessons, inputs=(diagnosis, decision))

        action = str(read_json(decision).get("action"))
        if action == "STOP":
            return self._command(run_dir, "finalize", "--cycle", cycle.name, output=run_dir / "terminal.json", inputs=(decision, registry))
        # The configured value is part of the immutable scientific input lock.
        # Goal-mode supervision may extend only this operational stopping budget
        # without rewriting the locked experiment protocol.
        configured_max_cycles = int(self.config.section("automation")["max_cycles"])
        if _cycle_budget_reached(configured_max_cycles, int(cycle.name)):
            return self._command(run_dir, "finalize", "--cycle", cycle.name, "--reason", "cycle_budget_exhausted", output=run_dir / "terminal.json", inputs=(decision, registry))
        next_cycle = cycle.parent / f"{int(cycle.name)+1:03d}"
        if action == "PROMOTE" and read_json(plan).get("evaluation_tier") == "development":
            return self._agent(
                "experiment_plan",
                (
                    selection,
                    run_dir / "inputs.lock.json",
                    decision,
                    plan,
                    implementation,
                    validation,
                ),
                next_cycle / "experiment_plan.json",
            )
        next_problem = next_cycle / "problem_diagnosis.json"
        inputs = self._problem_diagnosis_inputs(run_dir, cycle)
        return self._agent("problem_diagnosis", inputs, next_problem)
