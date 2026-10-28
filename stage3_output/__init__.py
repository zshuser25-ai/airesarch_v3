"""Export the measured final method without introducing another hypothesis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

FINAL_NAME = "final_method.md"


def _complete_control(path: Path, artifact_type: str) -> dict[str, Any]:
    from kvresearch.util import read_json, sha256_file

    value = read_json(path)
    if value.get("artifact_type") != artifact_type or value.get("status") != "complete":
        raise ValueError(f"尚无完整的 {artifact_type} 产物：{path}")
    for ref in value.get("inputs", []):
        source = Path(ref["path"])
        if not source.is_file() or sha256_file(source) != ref.get("sha256"):
            raise ValueError(f"输出证据缺失或哈希已变化：{source}")
    return value


def _reference(value: dict[str, Any], name: str, run_dir: Path) -> Path:
    from kvresearch.util import ensure_within

    matches = [ensure_within(Path(ref["path"]), run_dir)
               for ref in value.get("inputs", []) if Path(ref["path"]).name == name]
    if len(matches) != 1:
        raise ValueError(f"最终产物必须明确引用唯一的 {name}")
    return matches[0]


def _describe(value: Any) -> str:
    if isinstance(value, str):
        return value.strip() or "未提供。"
    if isinstance(value, list):
        return "\n".join(f"- {_describe(item)}" for item in value) or "未提供。"
    if isinstance(value, dict):
        return "\n".join(f"- **{key}**：{_describe(item)}" for key, item in value.items()) or "未提供。"
    return str(value) if value is not None else "未提供。"


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return "未提供"


def _baseline_score(baseline: dict[str, Any], task: str) -> Any:
    direct = baseline.get(task)
    if isinstance(direct, dict) and "score" in direct:
        return direct["score"]
    if task == "longbench" and isinstance(direct, dict):
        scores = [value for value in direct.values() if isinstance(value, (int, float))]
        return sum(scores) / len(scores) if scores else None
    return (baseline.get("math") or {}).get(task)


def export_method(try_dir: Path, run_dir: Path) -> Path:
    """Summarize only the candidate bound to the completed terminal artifact."""
    from kvresearch.util import atomic_write_text, ensure_within, read_jsonl

    try_dir, run_dir = try_dir.resolve(), ensure_within(run_dir, try_dir)
    preparation = try_dir / "preparation.md"
    if not preparation.is_file() or not (run_dir / "request.md").is_file():
        raise ValueError("缺少本次预准备材料或第二阶段研究请求")
    if preparation.read_bytes() != (run_dir / "request.md").read_bytes():
        raise ValueError("预准备材料与第二阶段锁定的请求不一致；请新建 try")
    terminal_path = run_dir / "terminal.json"
    if not terminal_path.is_file():
        raise ValueError("第二阶段尚未结束；暂不生成最终方法报告")
    terminal = _complete_control(terminal_path, "terminal")
    decision_path = _reference(terminal, "decision.json", run_dir)
    registry_path = _reference(terminal, "verified_registry.json", run_dir)
    if registry_path.parent != decision_path.parent:
        raise ValueError("最终决策与数值注册表不属于同一轮研究")
    cycle = decision_path.parent
    if cycle.parent != run_dir / "cycles" or not cycle.name.isdigit():
        raise ValueError("最终决策不在本次研究的 cycle 目录内")
    decision = _complete_control(decision_path, "decision")
    registry = _complete_control(registry_path, "verified_registry")
    metrics_path = _reference(registry, "metrics.json", run_dir)
    if metrics_path.parent != cycle:
        raise ValueError("最终数值注册表没有绑定同一轮指标")
    metrics = _complete_control(metrics_path, "metrics")
    plan = _complete_control(cycle / "experiment_plan.json", "experiment_plan")
    candidate_id = terminal.get("final_candidate_id")
    if not candidate_id or candidate_id != plan.get("candidate_id"):
        raise ValueError("最终候选与最终评测计划不一致")

    # Formal plans can bind a prior development selection. Follow that exact
    # provenance chain, rather than searching for a matching unbound idea.
    selection_path = _reference(plan, "selection.json", run_dir)
    selection = _complete_control(selection_path, "selection")
    if selection.get("selected_candidate_id") != candidate_id:
        raise ValueError("最终方法与评测计划引用的候选选择不一致")
    candidates_path = _reference(selection, "candidates.jsonl", run_dir)
    candidate = next((row for row in read_jsonl(candidates_path) if row.get("candidate_id") == candidate_id), None)
    if candidate is None:
        raise ValueError(f"找不到最终方法 {candidate_id} 的候选说明；不能生成方法内容")

    achieved = terminal.get("termination_type") == "objective_met" and decision.get("action") == "STOP"
    outcome = "已达到第二阶段的目标判定。" if achieved else "研究已结束，但未记录目标达成；下述为结束时的候选方法。"
    lines = [
        f"# {try_dir.name}：最终方法与主要思想", "",
        f"研究输入：[预准备材料](preparation.md)", "",
        f"研究状态：{outcome}", "",
        f"终止类型：`{terminal.get('termination_type')}`；最终决策：`{decision.get('action')}`。", "",
        "## 最终方法", "", f"候选标识：`{candidate_id}`", "",
        _describe(candidate.get("mechanism")), "",
        "## 主要思想", "", _describe(candidate.get("causal_chain")), "",
        "## 资源与预算设计", "", _describe(candidate.get("budget_design", candidate.get("storage_design"))), "",
        "## 实现要点", "", _describe(candidate.get("minimal_code_change")), "",
        "## 评测结果", "",
        f"评测范围：`{metrics.get('comparison_mode', plan.get('evaluation_tier'))}`。", "",
        "| 指标 | 候选 | 对比基线 |", "|---|---:|---:|",
    ]
    if "quality" in metrics:
        for name, metric in metrics["quality"].items():
            label = str(metric.get("label", name)).replace("|", "\\|")
            lines.append(f"| {label} | {_number(metric.get('candidate'))} | {_number(metric.get('baseline'))} |")
        constraint_text = f"本次声明的资源与预算约束：`{metrics.get('constraints_pass')}`。"
    else:
        baseline = metrics.get("baseline_reference") or {}
        for task, title in (("longbench", "LongBench 宏平均"), ("gsm8k", "GSM8K accuracy（%）"),
                            ("math500", "MATH500 accuracy（%）")):
            lines.append(f"| {title} | {_number((metrics.get(task) or {}).get('score'))} | {_number(_baseline_score(baseline, task))} |")
        constraint_text = f"理论存储检查：`{metrics.get('theoretical_storage_pass')}`；实际存储检查：`{metrics.get('actual_storage_pass')}`。"
    lines.extend([
        "", constraint_text, "",
        "## 结论与未解决问题", "", _describe(decision.get("rationale")), "",
        _describe(terminal.get("unresolved")) if terminal.get("unresolved") else "终态文件未列出未解决问题。", "",
        "## 证据与复现", "",
    ])
    for path in (candidates_path, selection_path, cycle / "experiment_plan.json", metrics_path, registry_path, decision_path, terminal_path):
        relative = path.relative_to(try_dir).as_posix()
        lines.append(f"- [{relative}]({relative})")
    lines.extend(["", "复现入口：", "", "```text", str(terminal.get("reproduction_entrypoint", "未提供")), "```", ""])
    output = try_dir / FINAL_NAME
    atomic_write_text(output, "\n".join(lines))
    return output
