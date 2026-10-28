from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kvresearch.navigator import Action
from kvresearch.util import atomic_write_text


_CURRENT_START = "<!-- kvresearch-current:start -->"
_CURRENT_END = "<!-- kvresearch-current:end -->"
_TIMELINE_END = "<!-- kvresearch-timeline:end -->"
_CURRENT_STATUS_RE = re.compile(r"^- 当前状态[：:].*$", re.MULTILINE)
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_SUMMARY_KEYS = (
    "status", "action", "measurement_valid", "rationale", "next_cycle_mode",
    "failures", "errors", "longbench", "gsm8k", "math500",
    "theoretical_storage_pass", "actual_storage_pass", "cooldown", "output",
    "task_packet", "attempts", "error", "error_code", "exit_code",
    "interruption", "timed_out", "oom", "message", "quality", "constraints_pass", "coverage",
)


def _argument_value(argv: tuple[str, ...], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _path_value(action: Action, name: str) -> str | None:
    candidates = [action.output or "", *action.inputs]
    pattern = rf"/{name}/([0-9]+)(?:/|$)"
    for candidate in candidates:
        match = re.search(pattern, candidate)
        if match:
            return match.group(1)
    return None


def _summary(outcome: Any) -> Any:
    if not isinstance(outcome, dict):
        return outcome
    selected = {key: outcome[key] for key in _SUMMARY_KEYS if key in outcome}
    if "agent_outcome" in outcome:
        selected["agent_outcome"] = _summary(outcome["agent_outcome"])
    if "command_outcome" in outcome:
        selected["command_outcome"] = _summary(outcome["command_outcome"])
    return selected or {"result_type": "dict", "keys": sorted(outcome)}


def _stage_label(step: str) -> str:
    return {
        "preflight": "执行预检", "monitor": "开发评测", "aggregate": "结果聚合",
        "experiment_diagnosis": "实验诊断", "decision": "自动决策",
        "lessons": "经验归档", "problem_diagnosis": "问题诊断",
        "candidates": "候选生成", "selection": "候选选择",
        "experiment_plan": "实验规划", "implementation": "实现",
        "implementation_repair": "实现修复", "execution_repair": "执行修复",
    }.get(step, step)


def _status_label(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "完成"
    if summary.get("action"):
        return str(summary["action"])
    status = str(summary.get("status", "complete"))
    return {"complete": "完成", "completed": "完成", "failed": "技术失败",
            "running": "进行中"}.get(status.lower(), status)


def _evidence_text(summary: Any, output: str | None) -> str:
    if isinstance(summary, dict):
        parts: list[str] = []
        for key in (
            "measurement_valid", "error_code", "interruption", "rationale",
            "longbench", "gsm8k", "math500", "theoretical_storage_pass",
            "actual_storage_pass", "cooldown", "quality", "constraints_pass", "coverage",
        ):
            if key in summary and summary[key] is not None:
                value = json.dumps(summary[key], ensure_ascii=False, sort_keys=True)
                parts.append(f"`{key}={value}`")
        if not parts:
            parts.append("阶段产物已形成并由流水线收口")
    else:
        parts = [str(summary)]
    if output:
        parts.append(f"证据：`{output}`")
    return "；".join(parts).replace("|", "\\|").replace("\n", " ")


def _artifact_details(action: Action) -> list[str]:
    """Return compact, authoritative details for artifacts with generic outcomes."""
    if not action.output:
        return []
    path = Path(action.output)
    if path.suffix != ".json" or not path.is_file():
        return []
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if action.step == "experiment_plan":
        datasets = artifact.get("datasets") or []
        budgets = artifact.get("budgets") or {}
        decoding = artifact.get("decoding") or {}
        storage = artifact.get("storage_accounting") or {}
        details = [
            f"tier={artifact.get('evaluation_tier')}",
            f"tasks={len(datasets)}",
            f"gpu_count={budgets.get('gpu_count')}",
            f"batch={budgets.get('batch_size_schedule')}",
            f"math_cap={decoding.get('math_max_total_length')}",
            f"theoretical_bits={storage.get('theoretical_average_bits')}",
        ]
        return [item for item in details if not item.endswith("=None")]
    if action.step in {"implementation", "implementation_manifest"}:
        details = [
            f"source_tree_sha256={artifact.get('source_tree_sha256')}",
            f"patch_hash={artifact.get('patch_hash')}",
            f"changed_files={len(artifact.get('changed_files') or [])}",
        ]
        return [item for item in details if not item.endswith("=None")]
    return []


def _new_report(run_dir: Path, current: str, row: str) -> str:
    created_at = datetime.now(_LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    return "\n".join([
        "# KVResearch 自动循环进度", "", f"- 项目：`{run_dir.parent.parent}`",
        f"- 运行：`{run_dir}`", "- 模式：Codex Goal",
        f"- 创建时间：{created_at}（Asia/Shanghai）",
        "- 报告机制：单文件维护；完整阶段形成正式产物后追加时间线，技术过程以可审计证据说明。",
        f"- 当前状态：{current}",
        "", "## 当前阶段", "", _CURRENT_START, current, _CURRENT_END, "",
        "## 已完成", "",
        "- 研究请求、输入锁、运行配置与阶段产物均由现有流水线持续校验。",
        "- 每条时间线记录均指向权威落盘产物；技术失败不冒充算法结论。",
        "", "## 时间线", "", "| 时间（Asia/Shanghai） | 阶段 | 状态 | 说明 |",
        "|---|---|---|---|", row, _TIMELINE_END, "", "## 关键路径", "",
        "```text", "准备材料 → 固定数据 → 文献与诊断 → 三个科研候选 → 选择 → 规划 → 隔离实现与验证",
        "→ 非评分冒烟 ⇄ 同候选技术修复 → paired开发评测 → 聚合与诊断",
        "→ 按本次指标与约束裁决 → 正式总量30% → 通过后补测65%及重跑开发5% → 全量决策", "```", "",
    ])


def write_stage_report(report_dir: Path, run_dir: Path, action: Action, outcome: Any) -> Path:
    """Maintain one human-readable progress report and append one timeline row."""
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(_LOCAL_TZ)
    cycle = _argument_value(action.argv, "--cycle") or _path_value(action, "cycles") or "run"
    attempt = _argument_value(action.argv, "--attempt") or _path_value(action, "attempts")
    output = report_dir / "kvresearch_stage_report.md"
    summary = _summary(outcome)
    scope = f"cycle{cycle} / {_stage_label(action.step)}"
    if attempt:
        scope += f" attempt{attempt}"
    status = _status_label(summary)
    evidence = _evidence_text(summary, action.output)
    details = _artifact_details(action)
    if details:
        evidence = f"{'；'.join(f'`{item}`' for item in details)}；{evidence}"
    current = f"{scope} 已收口，状态为 `{status}`。{evidence}"
    row = f"| {created_at.strftime('%Y-%m-%d %H:%M')} | {scope} | {status} | {evidence} |"
    if output.is_file():
        body = output.read_text(encoding="utf-8")
        if _CURRENT_START in body and _CURRENT_END in body and _TIMELINE_END in body:
            body = re.sub(rf"{re.escape(_CURRENT_START)}.*?{re.escape(_CURRENT_END)}",
                          f"{_CURRENT_START}\n{current}\n{_CURRENT_END}", body,
                          flags=re.DOTALL)
            if _CURRENT_STATUS_RE.search(body):
                body = _CURRENT_STATUS_RE.sub(f"- 当前状态：{current}", body, count=1)
            body = body.replace(_TIMELINE_END, f"{row}\n{_TIMELINE_END}", 1)
        else:
            body = _new_report(run_dir, current, row)
    else:
        body = _new_report(run_dir, current, row)
    atomic_write_text(output, body)
    return output
