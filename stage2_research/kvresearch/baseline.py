from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_baseline_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    longbench: dict[str, float] = {}
    math: dict[str, float] = {}
    section = ""
    for line in text.splitlines():
        if line.strip() == "### LongBench":
            section = "longbench"
        elif line.strip() == "### 数学任务":
            section = "math"
        if not line.startswith("|") or "---" in line or "数据集" in line:
            continue
        cells = [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]
        if section == "longbench" and len(cells) >= 3:
            try:
                longbench[cells[0]] = float(cells[2])
            except ValueError:
                pass
        elif section == "math" and len(cells) >= 2:
            try:
                math[cells[0].lower()] = float(cells[1].rstrip("%"))
            except ValueError:
                pass
    return {"longbench": longbench, "math": math}

