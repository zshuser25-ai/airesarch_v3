from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from kvresearch.config import load_config


LONG_DEV = ("qasper", "lcc", "gov_report")
RUNTIME_DATA_FORMAT_VERSION = 3
_SMOKE_ROLLOVER_CONTEXT = " ".join(["context"] * 192)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_smoke_datasets(
    config_path: str,
    run_dir: str,
    source_tier: str,
    smoke_key: str,
    tasks: list[str],
) -> Path:
    """Create a one-row-per-task runtime set for the implementation hard gate.

    The smoke set is derived only from the already locked evaluation partition.
    It is never scored and cannot be consumed by the metric aggregator.  Its
    purpose is to prove that the exact candidate entrypoint can load all of its
    artifacts and close one real POD inference row before the 32-row or formal
    evaluation is authorized.
    """
    config = load_config(config_path)
    run = Path(run_dir).resolve()
    if source_tier not in {"development", "formal"}:
        raise ValueError(f"unsupported smoke source tier: {source_tier}")
    if not smoke_key or any(part in smoke_key for part in ("/", "..")):
        raise ValueError("smoke_key must be a stable path component")
    target = run / "runtime_datasets" / "implementation_smoke" / smoke_key
    marker = target / ".complete.json"
    expected_tasks = sorted(set(tasks))
    if marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("format_version") == RUNTIME_DATA_FORMAT_VERSION and value.get("tasks") == expected_tasks:
            return target

    project = config.section("project")
    fixed = Path(project["fixed_testset" if source_tier == "development" else "formal_testset"])
    allowed = set(config.section("evaluation")["development_tasks"] if source_tier == "development" else
                  config.section("evaluation")["longbench_tasks"] + config.section("evaluation")["math_tasks"])
    if not expected_tasks or not set(expected_tasks).issubset(allowed):
        raise ValueError(f"invalid smoke task set for {source_tier}: {expected_tasks}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    from datasets import Dataset, DatasetDict

    manifest_tasks = {}
    long_tasks = set(config.section("evaluation")["longbench_tasks"])
    for task in expected_tasks:
        source = fixed / f"{task}.jsonl"
        rows = _read_jsonl(source)
        if not rows:
            raise RuntimeError(f"locked smoke source is empty: {source}")
        row = dict(rows[0])
        if task in long_tasks:
            DatasetDict({"test": Dataset.from_list([row])}).save_to_disk(str(target / task))
            output_name = task
        else:
            # Math prompts in the locked set are often shorter than the
            # 128-token exact residual.  A liveness smoke that never rolls K
            # into the compressed history does not exercise a KV candidate.
            # Extend only the non-scoring runtime copy, identically for both
            # arms; the locked source row and all scored datasets stay intact.
            prompt_field = "problem" if task == "math500" else "question"
            original_prompt = row.get(prompt_field)
            if not isinstance(original_prompt, str) or not original_prompt.strip():
                raise RuntimeError(f"smoke source lacks {prompt_field}: {source}")
            row[prompt_field] = (
                original_prompt
                + "\n\n[NON-SCORING KV ROLLOVER SMOKE CONTEXT]\n"
                + _SMOKE_ROLLOVER_CONTEXT
            )
            output_name = "math.jsonl" if task == "math500" else "gsm8k.jsonl"
            (target / output_name).write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        manifest_tasks[task] = {
            "rows": 1,
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fixed_sample": row.get("_fixed_sample"),
            "forced_compressed_history_smoke": task not in long_tasks,
        }
    marker.write_text(
        json.dumps(
            {
                "tier": "implementation_smoke",
                "source_tier": source_tier,
                "status": "complete",
                "format_version": RUNTIME_DATA_FORMAT_VERSION,
                "tasks": expected_tasks,
                "manifest": {"samples_per_task": 1, "tasks": manifest_tasks},
                "non_scoring": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def prepare_datasets(config_path: str, run_dir: str, tier: str) -> Path:
    config = load_config(config_path)
    run = Path(run_dir).resolve()
    target = run / "runtime_datasets" / tier
    marker = target / ".complete.json"
    if marker.is_file():
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            marker_value = {}
        if marker_value.get("format_version") == RUNTIME_DATA_FORMAT_VERSION:
            return target
    target.mkdir(parents=True, exist_ok=True)
    project = config.section("project")
    setting = {"development": "fixed_testset", "formal": "formal_testset",
               "formal_remaining": "formal_remaining_testset"}.get(tier)
    if setting is None:
        raise ValueError(f"unknown evaluation tier: {tier}")
    fixed = Path(project[setting])
    if tier == "development":
        from datasets import Dataset, DatasetDict

        for task in LONG_DEV:
            rows = [json.loads(line) for line in (fixed / f"{task}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            output = target / task
            if output.exists():
                shutil.rmtree(output)
            DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(output))
    elif tier in {"formal", "formal_remaining"}:
        from datasets import Dataset, DatasetDict

        for task in config.section("evaluation")["longbench_tasks"]:
            rows = [json.loads(line) for line in (fixed / f"{task}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            destination = target / task
            if destination.exists():
                shutil.rmtree(destination)
            DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(destination))
    else:
        raise ValueError(f"unknown evaluation tier: {tier}")
    shutil.copy2(fixed / "gsm8k.jsonl", target / "gsm8k.jsonl")
    shutil.copy2(fixed / "math500.jsonl", target / "math.jsonl")
    manifest = json.loads((fixed / "manifest.json").read_text(encoding="utf-8"))
    marker.write_text(
        json.dumps(
            {
                "tier": tier,
                "status": "complete",
                "format_version": RUNTIME_DATA_FORMAT_VERSION,
                "manifest": manifest,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tier", choices=("development", "formal", "formal_remaining"))
    parser.add_argument("--smoke-source-tier", choices=("development", "formal"))
    parser.add_argument("--smoke-key")
    parser.add_argument("--tasks")
    args = parser.parse_args(argv)
    if args.smoke_source_tier:
        if not args.smoke_key or not args.tasks:
            parser.error("--smoke-source-tier requires --smoke-key and --tasks")
        print(prepare_smoke_datasets(args.config, args.run, args.smoke_source_tier,
                                     args.smoke_key, args.tasks.split(",")))
    elif args.tier:
        print(prepare_datasets(args.config, args.run, args.tier))
    else:
        parser.error("either --tier or --smoke-source-tier is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
