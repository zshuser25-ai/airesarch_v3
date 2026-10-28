from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kvresearch.agent import run_agent_action
from kvresearch.config import DEFAULT_CONFIG, Config, load_config
from kvresearch.executor import execute, monitor
from kvresearch.initialize import initialize_run
from kvresearch.navigator import Action, Navigator
from kvresearch.pipeline import aggregate, decide, diagnose, finalize, learn, validate_cycle, verify
from kvresearch.reporting import write_stage_report
from kvresearch.remote import RemotePod
from kvresearch.schema import ArtifactValidator
from kvresearch.workspace import prepare_workspace, snapshot_implementation


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _run_path(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _dispatch(config: Config, args: argparse.Namespace) -> Any:
    command = args.command
    from kvresearch.research import enabled
    if enabled(config):
        from kvresearch.research import initialize, next_action, perform, run_loop
        if command == "init":
            request = Path(args.request_file).read_text(encoding="utf-8") if args.request_file else args.request
            if not args.run_id:
                raise ValueError("通用研究初始化需要明确的 --run-id；推荐使用 workflow.py start/run")
            preparation = Path(args.request_file).resolve() if args.request_file else None
            return {"run_dir": str(initialize(config, request, args.run_id, preparation))}
        run = _run_path(args.run) if getattr(args, "run", None) else None
        if run is None:
            raise ValueError("本步骤需要 --run；请先通过 workflow.py start/run 创建准备材料与研究运行")
        if command == "next":
            return next_action(config, run).to_dict()
        if command == "auto":
            return run_loop(config, run, args.max_steps, Path(args.report_dir) if args.report_dir else run.parent.parent / "reports")
        action = next_action(config, run)
        if command == "agent-task":
            return perform(config, run, action) if action.kind == "agent" else {"status": "not_an_agent_step", "action": action.to_dict()}
        if command != action.step:
            raise ValueError(f"当前下一步为 {action.step}，请按导航顺序执行；硬件与材料由准备文件解析")
        return perform(config, run, action)
    if command == "init":
        request = Path(args.request_file).read_text(encoding="utf-8") if args.request_file else args.request
        return {"run_dir": str(initialize_run(config, request, args.run_id, args.parent_run_id))}
    run = _run_path(args.run) if hasattr(args, "run") and args.run else None
    if command == "next":
        return Navigator(config).next(run).to_dict()
    if command == "validate-file":
        validator = ArtifactValidator(config.catalog_path)
        result = validator.validate_jsonl(Path(args.file), args.artifact_type) if args.jsonl else validator.validate_control(Path(args.file), args.artifact_type)
        return {"valid": result.valid, "errors": list(result.errors)}
    if command == "prepare-workspace":
        return {"workspace": str(prepare_workspace(config, run, args.cycle))}
    if command == "snapshot-implementation":
        return snapshot_implementation(config, run, args.cycle, args.candidate_id)
    if command == "preflight":
        return RemotePod(config).preflight(run, args.cycle, args.attempt)
    if command == "validate-cycle":
        return validate_cycle(config, run, args.cycle)
    if command == "execute":
        return execute(config, run, args.cycle, args.attempt)
    if command == "monitor":
        return monitor(config, run, args.cycle, args.attempt)
    if command == "aggregate":
        return aggregate(config, run, args.cycle)
    if command == "verify":
        return verify(config, run, args.cycle)
    if command == "diagnose":
        return diagnose(config, run, args.cycle)
    if command == "decide":
        return decide(config, run, args.cycle)
    if command == "learn":
        return {"lessons": learn(run, args.cycle)}
    if command == "finalize":
        return finalize(run, args.cycle, args.reason)
    if command == "agent-task":
        action = Navigator(config).next(run)
        if action.kind != "agent":
            return {"status": "not_an_agent_step", "action": action.to_dict()}
        return run_agent_action(config, run, action)
    if command == "environment-check":
        snapshot = RemotePod(config).environment_snapshot()
        output = config.project_root / "environment" / "current_pod.json"
        from kvresearch.util import atomic_write_json
        atomic_write_json(output, snapshot)
        return {"output": str(output), **snapshot}
    if command == "auto":
        navigator = Navigator(config)
        history = []
        for _ in range(args.max_steps):
            action = navigator.next(run)
            history.append(action.to_dict())
            if action.kind in {"terminal", "repair_input"}:
                break
            if action.kind == "agent":
                outcome = run_agent_action(config, run, action)
                history.append({"agent_outcome": outcome})
                if outcome.get("status") != "complete":
                    break
                if args.report_dir:
                    history.append({"stage_report": str(write_stage_report(Path(args.report_dir), run, action, outcome))})
                continue
            cmd_args = list(action.argv)
            if cmd_args and cmd_args[0] == "kvresearch":
                cmd_args = cmd_args[1:]
            nested = build_parser().parse_args(cmd_args)
            outcome = _dispatch(config, nested)
            history.append({"command_outcome": outcome})
            if args.report_dir:
                history.append({"stage_report": str(write_stage_report(Path(args.report_dir), run, action, outcome))})
        return {"history": history, "final": navigator.next(run).to_dict()}
    raise ValueError(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvresearch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    source = init.add_mutually_exclusive_group(required=True)
    source.add_argument("--request")
    source.add_argument("--request-file")
    init.add_argument("--run-id")
    init.add_argument("--parent-run-id")
    for name in ("next", "agent-task", "environment-check"):
        item = sub.add_parser(name)
        if name != "environment-check":
            item.add_argument("--run", required=True)
    freeze = sub.add_parser("freeze-data")
    freeze.add_argument("--run", required=True)
    validate_file = sub.add_parser("validate-file")
    validate_file.add_argument("--file", required=True)
    validate_file.add_argument("--artifact-type", required=True)
    validate_file.add_argument("--jsonl", action="store_true")
    for name in ("prepare-workspace", "validate-cycle", "aggregate", "verify", "diagnose", "decide", "learn"):
        item = sub.add_parser(name)
        item.add_argument("--run", required=True)
        item.add_argument("--cycle", required=True)
    snapshot = sub.add_parser("snapshot-implementation")
    snapshot.add_argument("--run", required=True)
    snapshot.add_argument("--cycle", required=True)
    snapshot.add_argument("--candidate-id", required=True)
    for name in ("preflight", "execute", "monitor"):
        item = sub.add_parser(name)
        item.add_argument("--run", required=True)
        item.add_argument("--cycle", required=True)
        item.add_argument("--attempt", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--run", required=True)
    finish.add_argument("--cycle", required=True)
    finish.add_argument("--reason")
    auto = sub.add_parser("auto")
    auto.add_argument("--run", required=True)
    auto.add_argument("--max-steps", type=int, default=100)
    auto.add_argument("--report-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = _dispatch(config, args)
        _print(result)
        if isinstance(result, dict) and result.get("valid") is False:
            return 1
        return 0
    except Exception as exc:
        _print({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
