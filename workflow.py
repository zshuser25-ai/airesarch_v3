"""Three-stage entrypoint with preparation-driven research and frozen data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = PROJECT_ROOT / "stage2_research"
sys.path.insert(0, str(RESEARCH_ROOT))

from stage1_preparation import PREPARATION_NAME, collect_inputs, create_try, resolve_try


def run_research(try_dir: Path, config_source: Path, max_steps: int = 100) -> dict[str, Any]:
    # Import the GPU/Agent framework only after preparation has finished.
    import yaml

    from kvresearch.agent import run_agent_action
    from kvresearch.cli import _dispatch, build_parser as research_parser
    from kvresearch.config import load_config
    from kvresearch.initialize import build_inputs_lock, initialize_run
    from kvresearch.navigator import Navigator
    from kvresearch.reporting import write_stage_report
    from kvresearch.util import atomic_write_text
    from stage3_output import export_method

    if max_steps < 1:
        raise ValueError("max-steps 必须大于零")
    preparation = try_dir / PREPARATION_NAME
    request = preparation.read_text(encoding="utf-8")
    if not request.strip():
        raise ValueError("预准备材料不能为空")
    config_path = try_dir / "config.kvresearch.yaml"
    if not config_path.is_file():
        source = load_config(config_source)
        data = copy.deepcopy(source.data)
        data["project"]["runs_dir"] = str(try_dir / "runs")
        atomic_write_text(config_path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    config = load_config(config_path)
    if config.runs_dir != try_dir / "runs":
        raise ValueError("本次配置的 runs_dir 必须位于当前 try 的 runs/ 目录")
    run_dir = config.runs_dir / try_dir.name
    from kvresearch.research import enabled, initialize, run_loop
    if enabled(config):
        if not run_dir.exists():
            initialize(config, request, try_dir.name, preparation)
        if (run_dir / "request.md").read_text(encoding="utf-8") != request:
            raise ValueError("预准备材料与本次锁定请求不一致；请新建 try")
        return {"try_dir": str(try_dir), **run_loop(config, run_dir, max_steps, try_dir / "reports")}
    if not run_dir.exists():
        # The legacy initializer creates its directory before checking inputs.
        # Read-check them first so missing materials leave preparation usable.
        build_inputs_lock(config, try_dir.name, preparation)
        initialize_run(config, request, run_id=try_dir.name)
    if not (run_dir / "request.md").is_file() or (run_dir / "request.md").read_text(encoding="utf-8") != request:
        raise ValueError("预准备材料与本次锁定请求不一致；请新建 try")

    navigator = Navigator(config)
    completed_steps = 0
    for _ in range(max_steps):
        action = navigator.next(run_dir)
        if action.kind in {"terminal", "repair_input"}:
            break
        if action.kind == "agent":
            # This outer handoff adds the original Markdown to every Agent
            # task without changing any research prompt, gate, or decision.
            inputs = (str(preparation),) + tuple(path for path in action.inputs if path != str(preparation))
            action = replace(action, inputs=inputs)
            outcome = run_agent_action(config, run_dir, action)
            if outcome.get("status") != "complete":
                return {"state": "AGENT_INCOMPLETE", "step": action.step, "run_dir": str(run_dir),
                        "completed_steps": completed_steps, "outcome": outcome}
        else:
            command = list(action.argv)
            if command and command[0] == "kvresearch":
                command = command[1:]
            outcome = _dispatch(config, research_parser().parse_args(command))
        write_stage_report(try_dir / "reports", run_dir, action, outcome)
        completed_steps += 1
        print(f"第二阶段：{action.step} 已执行。", flush=True)

    final = navigator.next(run_dir)
    result = {"try_dir": str(try_dir), "run_dir": str(run_dir), "state": final.state,
              "step": final.step, "completed_steps": completed_steps, "notes": final.notes}
    if final.state == "DONE":
        result["final_method"] = str(export_method(try_dir, run_dir))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预准备 → 研究执行 → 方法输出")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="收集四项研究定义，创建下一个 try 目录")
    start.add_argument("--goal-baseline", help="科研目标与对比基线")
    start.add_argument("--model", help="测试使用模型")
    start.add_argument("--datasets", help="数据集地址")
    start.add_argument("--hardware", help="可用硬件")
    run = sub.add_parser("run", help="读取已有 try 的准备材料，执行或恢复第二阶段")
    run.add_argument("--try", dest="try_name", required=True)
    run.add_argument("--config", type=Path, default=RESEARCH_ROOT / "config.kvresearch.yaml",
                     help="首次执行时复制的第二阶段配置模板；恢复时读取 try 内的配置")
    run.add_argument("--max-steps", type=int, default=100)
    output = sub.add_parser("output", help="从已经完成的研究产物重新生成最终方法 Markdown")
    output.add_argument("--try", dest="try_name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv or ["start"])
    try:
        if args.command == "start":
            values = collect_inputs({"goal_baseline": args.goal_baseline, "model": args.model,
                                     "datasets": args.datasets, "hardware": args.hardware})
            try_dir = create_try(PROJECT_ROOT, values)
            result = {"stage": "preparation_complete", "try_dir": str(try_dir),
                      "preparation": str(try_dir / PREPARATION_NAME),
                      "next": f"python3 workflow.py run --try {try_dir.name}"}
        elif args.command == "run":
            result = run_research(resolve_try(PROJECT_ROOT, args.try_name), args.config, args.max_steps)
        else:
            from stage3_output import export_method
            try_dir = resolve_try(PROJECT_ROOT, args.try_name)
            result = {"final_method": str(export_method(try_dir, try_dir / "runs" / try_dir.name))}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("state") in {"INVALID", "AGENT_INCOMPLETE", "NEEDS_INPUT"} else 0
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc) or "输入或执行已中断"
        print(json.dumps({"error": type(exc).__name__, "message": message}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
