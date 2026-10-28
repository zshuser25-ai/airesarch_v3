"""Collect the user's research definition before starting any experiment."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


FIELDS = (
    ("goal_baseline", "科研目标与对比基线", "研究目标、对照方法或代码地址、比较指标及预算"),
    ("model", "测试使用模型", "模型名称、版本及本地或远端地址"),
    ("datasets", "数据集地址", "原始数据入口、任务和格式；第二阶段固定开发5%、正式30%、补测65%"),
    ("hardware", "可用硬件", "GPU 型号、数量、显存、节点地址及可用运行环境"),
)
TRY_NAME = re.compile(r"try([0-9]{3,})\Z")
PREPARATION_NAME = "preparation.md"


def collect_inputs(
    provided: Mapping[str, str | None],
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
) -> dict[str, str]:
    """Prompt for every missing field; finish a multiline answer with a dot."""
    if input_fn is None:
        input_fn = input
    values: dict[str, str] = {}
    for key, title, hint in FIELDS:
        value = provided.get(key)
        if value is not None and value.strip():
            values[key] = value.strip()
            continue
        while True:
            output_fn(f"\n请定义：{title}\n{hint}\n可输入多行，以单独一行 . 结束。此项必填。")
            lines = []
            try:
                while True:
                    line = input_fn("> ")
                    if line.strip() == ".":
                        break
                    lines.append(line)
            except EOFError as exc:
                raise ValueError(f"{title}尚未填写完成；没有创建新的 try 目录") from exc
            value = "\n".join(lines).strip()
            if value:
                values[key] = value
                break
            output_fn(f"{title}不能为空，请重新输入。")
    return values


def create_try(project_root: Path, values: Mapping[str, str]) -> Path:
    """Allocate a new monotonically numbered directory after all inputs exist."""
    normalized = {}
    for key, title, _ in FIELDS:
        value = values.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"缺少必填内容：{title}")
        normalized[key] = value.strip()
    project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    while True:
        indices = [int(match.group(1)) for path in project_root.iterdir()
                   if (match := TRY_NAME.fullmatch(path.name))]
        try_dir = project_root / f"try{max(indices, default=0) + 1:03d}"
        try:
            try_dir.mkdir()
            break
        except FileExistsError:
            # mkdir, rather than a read-then-write counter, arbitrates races.
            continue
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"# {try_dir.name}：预准备阶段", "", f"创建时间：{created}", ""]
    for key, title, _ in FIELDS:
        lines.extend([f"## {title}", "", normalized[key], ""])
    lines.extend([
        "## 交给第二阶段的说明", "",
        "以上四项为本次用户定义的研究输入。第二阶段 Agent 必须先阅读本文原文，再开展研究。",
        "第二阶段先解析本次模型、基线、数据与硬件，再固定总量5%开发、30%正式、剩余65%补测。",
        "各阶段读取本文件；不预设其他项目的材料与评测协议。", "",
    ])
    fd, temporary = tempfile.mkstemp(prefix=".preparation-", suffix=".tmp", dir=try_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, try_dir / PREPARATION_NAME)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return try_dir


def resolve_try(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if path.parent != project_root.resolve() or not TRY_NAME.fullmatch(path.name):
        raise ValueError("研究目录必须是项目根目录下的 try001、try002 等目录")
    if not path.is_dir() or not (path / PREPARATION_NAME).is_file():
        raise ValueError(f"预准备材料不存在：{path / PREPARATION_NAME}")
    return path
