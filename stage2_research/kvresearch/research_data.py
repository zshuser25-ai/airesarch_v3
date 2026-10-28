"""Resolve user-defined dataset entrances and freeze disjoint 5/30/65 splits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from kvresearch.config import Config
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl, atomic_write_text, path_ref, read_json, read_jsonl, sha256_file


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PARTITIONS = ("development", "formal", "remaining")


def checked_control(path: Path, kind: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    if kind and value.get("artifact_type") != kind:
        raise ValueError(f"产物类型错误：{path}")
    if value.get("status") != "complete":
        raise ValueError(f"产物未完成：{path}")
    for ref in value.get("inputs", []):
        source = Path(ref["path"])
        if not source.is_file() or sha256_file(source) != ref.get("sha256"):
            raise ValueError(f"输入缺失或已变化：{source}")
    return value


def validate_context(value: dict[str, Any]) -> None:
    if value.get("ready") is False and value.get("missing_inputs"):
        return
    for key in ("goal", "model", "baseline", "datasets", "hardware", "metrics", "evaluation_protocol", "constraints"):
        if key not in value:
            raise ValueError(f"研究定义缺少 {key}；必须从 preparation.md 解析")
    if not isinstance(value["goal"], str) or not value["goal"].strip():
        raise ValueError("科研目标不能为空")
    for key in ("model", "baseline", "hardware", "evaluation_protocol"):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} 必须为对象")
    if not value["model"].get("name") or not value["baseline"].get("name"):
        raise ValueError("必须记录用户指定的模型和对比基线名称")
    datasets = value["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("必须解析至少一个数据集入口")
    ids = []
    for dataset in datasets:
        task = dataset.get("id", "")
        if not isinstance(task, str) or not IDENTIFIER.fullmatch(task):
            raise ValueError(f"无效数据集标识：{task}")
        source = dataset.get("source", {})
        if source.get("format") not in {"jsonl", "json", "csv", "parquet", "hf_disk", "hf_hub"}:
            raise ValueError(f"数据集 {task} 需要可执行的 JSONL/JSON/CSV/Parquet/HuggingFace 入口")
        if source["format"] == "hf_hub":
            if not source.get("repo") or not source.get("split"):
                raise ValueError("HuggingFace Hub 入口需要 repo 和 split")
        elif not Path(source.get("path", "")).is_absolute():
            raise ValueError(f"数据集 {task} 的本地路径必须为绝对路径")
        ids.append(task)
    if len(set(ids)) != len(ids):
        raise ValueError("数据集标识不能重复")
    metrics = value["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("必须固定用户目标对应的评测指标")
    metric_ids = []
    for metric in metrics:
        if not IDENTIFIER.fullmatch(metric.get("id", "")):
            raise ValueError("评测指标需要稳定的 id")
        if metric.get("direction") not in {"maximize", "minimize"}:
            raise ValueError("指标 direction 必须是 maximize 或 minimize")
        if metric.get("aggregation", "macro") not in {"macro", "micro"}:
            raise ValueError("指标 aggregation 必须是 macro 或 micro")
        if not metric.get("tasks") or not set(metric["tasks"]).issubset(ids):
            raise ValueError("指标 tasks 必须引用已经声明的数据集")
        metric_ids.append(metric["id"])
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("评测指标 id 不能重复")
    if not isinstance(value["constraints"], list):
        raise ValueError("constraints 必须是数组；无额外预算限制时填写 []")
    constraint_ids = []
    for item in value["constraints"]:
        if not IDENTIFIER.fullmatch(item.get("id", "")) or item.get("operator") not in {"<=", ">=", "=="}:
            raise ValueError("约束需要稳定 id 及 <=、>= 或 == 运算符")
        threshold = item.get("limit", item.get("baseline_ratio"))
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
            raise ValueError("约束必须提供有限数值 limit 或 baseline_ratio")
        constraint_ids.append(item["id"])
    if len(set(constraint_ids)) != len(constraint_ids):
        raise ValueError("约束 id 不能重复")
    hardware = value["hardware"]
    if hardware.get("mode", "local") not in {"local", "ssh"}:
        raise ValueError("硬件 mode 需要是 local 或 ssh")
    if hardware.get("mode") == "ssh" and (not hardware.get("host") or not hardware.get("user")):
        raise ValueError("SSH 硬件需要用户给定的 host 和 user")
    gpu_ids = hardware.get("gpu_ids", [])
    if not isinstance(gpu_ids, list) or len({str(gpu) for gpu in gpu_ids}) != len(gpu_ids):
        raise ValueError("gpu_ids 必须是无重复的设备列表；CPU 可使用 []")
    rule = value.get("success_rule", "any_gain_no_regressions")
    if rule not in {"any_gain_no_regressions", "all_strict"}:
        raise ValueError("success_rule 必须是 any_gain_no_regressions 或 all_strict")
    for resource in value.get("resources", []):
        if not isinstance(resource, dict) or not resource.get("role") or not Path(resource.get("path", "")).is_absolute():
            raise ValueError("额外材料应声明 role 与绝对 path；只声明本次科研需要的材料")


def _source_rows(source: dict[str, Any], cache_dir: Path) -> tuple[Iterable[Any], list[dict[str, str]]]:
    fmt = source["format"]
    if fmt in {"jsonl", "json", "csv", "parquet"}:
        path = Path(source["path"])
        refs = [path_ref(path)]
        if fmt == "jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif fmt == "json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            if source.get("records_key"):
                for key in source["records_key"].split("."):
                    rows = rows[key]
            if not isinstance(rows, list):
                raise ValueError(f"JSON 数据需要数组或正确的 records_key：{path}")
        elif fmt == "csv":
            with path.open(encoding=source.get("encoding", "utf-8"), newline="") as handle:
                rows = list(csv.DictReader(handle))
        else:
            from datasets import load_dataset
            rows = load_dataset("parquet", data_files=str(path), split="train", cache_dir=str(cache_dir))
        return rows, refs
    from datasets import load_dataset, load_from_disk
    if fmt == "hf_disk":
        path = Path(source["path"])
        dataset = load_from_disk(str(path))
        if source.get("split"):
            dataset = dataset[source["split"]]
        refs = [path_ref(file) for file in sorted(path.rglob("*")) if file.is_file()]
    else:
        dataset = load_dataset(source["repo"], name=source.get("subset"), split=source["split"],
                               revision=source.get("revision"), cache_dir=str(cache_dir))
        refs = []
    return dataset, refs


def freeze_datasets(config: Config, run_dir: Path) -> dict[str, Any]:
    lock_path = run_dir / "datasets.lock.json"
    if lock_path.is_file():
        return checked_control(lock_path, "datasets_lock")
    context_path = run_dir / "research_context.json"
    context = checked_control(context_path, "research_context")
    validate_context(context)
    fractions = config.section("dataset_split")
    requested = tuple(float(fractions[key]) for key in ("development_fraction", "formal_fraction", "remaining_fraction"))
    if any(abs(a - b) > 1e-12 for a, b in zip(requested, (0.05, 0.30, 0.65))):
        raise ValueError("本次协议固定开发占总量5%、正式占总量30%、补测占总量65%")
    seed = int(fractions.get("seed", 42))
    refs = [path_ref(context_path), path_ref(config.path), path_ref(run_dir / "request.md")]
    tasks = {}
    for dataset in context["datasets"]:
        task = dataset["id"]
        rows, source_refs = _source_rows(dataset["source"], run_dir / "dataset_cache")
        normalized = []
        seen_payloads: dict[str, int] = {}
        for index, row in enumerate(rows):
            payload = row if isinstance(row, dict) else {"value": row}
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest = hashlib.sha256(serialized.encode()).hexdigest()
            occurrence = seen_payloads.get(digest, 0)
            seen_payloads[digest] = occurrence + 1
            sample_id = f"{task}:{digest}:{occurrence}"
            normalized.append({"sample_id": sample_id, "task": task, "source_index": index, "data": payload})
        if len(normalized) < 3:
            raise ValueError(f"数据集 {task} 少于3条，不能形成三个非空互斥分区")
        ranked = sorted(normalized, key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
        dev_count = (len(ranked) * 5 + 99) // 100
        formal_count = (len(ranked) * 30 + 99) // 100
        if dev_count + formal_count >= len(ranked):
            raise ValueError(f"数据集 {task} 无法形成非空补测分区")
        partitions = {"development": ranked[:dev_count], "formal": ranked[dev_count:dev_count + formal_count],
                      "remaining": ranked[dev_count + formal_count:]}
        full_path = run_dir / "datasets" / "full" / f"{task}.jsonl"
        atomic_write_jsonl(full_path, normalized)
        refs.extend(source_refs + [path_ref(full_path)])
        entry = {"source": dataset["source"], "total_rows": len(normalized), "full": path_ref(full_path), "partitions": {}}
        all_ids = set()
        for partition, samples in partitions.items():
            path = run_dir / "datasets" / partition / f"{task}.jsonl"
            # Publish each frozen partition in original order for stable evaluation.
            samples = sorted(samples, key=lambda row: row["source_index"])
            atomic_write_jsonl(path, samples)
            ids = [row["sample_id"] for row in samples]
            assert not all_ids.intersection(ids)
            all_ids.update(ids)
            entry["partitions"][partition] = {**path_ref(path), "rows": len(samples), "sample_ids": ids}
            refs.append(path_ref(path))
        assert all_ids == {row["sample_id"] for row in normalized}
        tasks[task] = entry
    # Only the resources discovered from this preparation are required.
    for resource in context.get("resources", []):
        path = Path(resource["path"])
        if not path.exists():
            if resource.get("required", True):
                raise FileNotFoundError(f"本次科研材料尚未就绪：{resource['role']}：{path}")
            continue
        if path.is_file():
            refs.append(path_ref(path))
        elif resource.get("identity_files"):
            refs.extend(path_ref(path / name) for name in resource["identity_files"])
    artifact = control_envelope("datasets_lock", run_dir.name, "kvresearch.research_data", refs,
                                seed=seed, fractions={"development": 0.05, "formal": 0.30, "remaining": 0.65},
                                rounding="ceil development and formal counts; remaining is exact complement",
                                tasks=tasks, disjoint=True, full_coverage=True)
    atomic_write_json(lock_path, artifact)
    lines = ["# 第二阶段：固定数据划分", "", "各数据集按总量划分为开发5%、正式30%、补测65%；样本不足整除时开发和正式向上取整，补测取精确补集。", "",
             f"随机种子：{seed}。本次研究后续各轮复用相同ID与哈希，不重新抽样。", "",
             "| 数据集 | 总量 | 开发 | 正式 | 补测 |", "|---|---:|---:|---:|---:|"]
    for task, entry in tasks.items():
        counts = entry["partitions"]
        lines.append(f"| {task} | {entry['total_rows']} | {counts['development']['rows']} | {counts['formal']['rows']} | {counts['remaining']['rows']} |")
    atomic_write_text(run_dir.parent.parent / "dataset_split.md", "\n".join(lines) + "\n")
    return artifact
