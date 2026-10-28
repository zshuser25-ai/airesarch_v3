from __future__ import annotations

import copy
import json
from pathlib import Path

from kvresearch.config import Config, load_config
from kvresearch.schema import control_envelope
from kvresearch.util import atomic_write_json, atomic_write_jsonl


def test_config(tmp: Path) -> Config:
    base = load_config(Path(__file__).parent / "fixtures" / "legacy_vecinfer_config.yaml")
    data = copy.deepcopy(base.data)
    project = tmp / "project"
    project.mkdir()
    fixed = project / "fixed_small_testset"
    fixed.mkdir()
    tasks = {}
    for task in ("qasper", "lcc", "gov_report", "gsm8k", "math500"):
        path = fixed / f"{task}.jsonl"
        path.write_text("".join(json.dumps({"task": task, "sample_idx": index}) + "\n" for index in range(32)), encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks[task] = {"output": {"path": path.name, "rows": 32, "sha256": digest}}
    (fixed / "manifest.json").write_text(json.dumps({"samples_per_task": 32, "tasks": tasks}), encoding="utf-8")
    formal = project / "formal_30pct_testset"
    formal.mkdir()
    formal_tasks = {}
    all_formal = list(base.section("evaluation")["longbench_tasks"]) + list(base.section("evaluation")["math_tasks"])
    for task in all_formal:
        path = formal / f"{task}.jsonl"
        path.write_text("".join(json.dumps({"task": task, "sample_idx": index, "_fixed_sample": {"source_index": index, "source_id": f"{task}-{index}"}}) + "\n" for index in range(3)), encoding="utf-8")
        import hashlib
        formal_tasks[task] = {"output": {"path": path.name, "rows": 3, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (formal / "manifest.json").write_text(json.dumps({"sample_fraction": 0.3, "tasks": formal_tasks}), encoding="utf-8")
    formal_baseline = formal / "vecinfer_1bit_baseline.json"
    formal_baseline.write_text(json.dumps({"longbench": {"score": 45.5}, "gsm8k": {"score": 0.0},
                                           "math500": {"score": 2.8}}), encoding="utf-8")
    remaining = project / "formal_remaining_70pct_testset"
    remaining.mkdir()
    remaining_tasks = {}
    for task in all_formal:
        path = remaining / f"{task}.jsonl"
        path.write_text("".join(json.dumps({"task": task, "sample_idx": index, "_fixed_sample": {"source_index": index + 3, "source_id": f"{task}-{index + 3}"}}) + "\n" for index in range(7)), encoding="utf-8")
        import hashlib
        remaining_tasks[task] = {"output": {"path": path.name, "rows": 7, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (remaining / "manifest.json").write_text(json.dumps({"sample_fraction": 0.7, "tasks": remaining_tasks}), encoding="utf-8")
    baseline_report = project / "VECINFER_1BIT_BASELINE.md"
    baseline_report.write_text(
        "### LongBench\n| 数据集 | 样本数 | 1-bit | FP16 |\n|---|---:|---:|---:|\n| qasper | 200 | 34.5 | 46 |\n| lcc | 500 | 50.3 | 63 |\n| gov_report | 200 | 18.0 | 34 |\n| 宏平均 | 3150 | 45.5 | 53 |\n### 数学任务\n| 数据集 | 1-bit | FP16 |\n|---|---:|---:|\n| GSM8K | 0.0% | 82% |\n| MATH500 | 2.8% | 47% |\n",
        encoding="utf-8",
    )
    source = tmp / "vecinfer"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "x.py").write_text("x = 1\n", encoding="utf-8")
    # Host manifest tests need file identities, not trained codebook tensors.
    # Keep the entire fixture local so this suite needs no external assets.
    centroids = source / "centroids"
    centroids.mkdir()
    storage = data["storage"]["vecinfer"]
    for layer in range(storage["layers"]):
        (centroids / f"key_cent_layer{layer}_{storage['key_subspaces']}_{storage['key_bits_per_code']}.pq.pt").write_bytes(f"fixture-k-{layer}".encode())
        (centroids / f"val_cent_layer{layer}_{storage['value_subspaces']}_{storage['value_bits_per_code']}.pq.pt").write_bytes(f"fixture-v-{layer}".encode())
    secondary = tmp / "vecinfer_v1"
    secondary.mkdir()
    model = tmp / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    model_cfg = tmp / "model.json"
    model_cfg.write_text(json.dumps({"model_name": "llama", "folder": str(model)}), encoding="utf-8")
    data["project"].update(
        root=str(project), runs_dir=str(project / "runs"), baseline_report=str(baseline_report),
        fixed_testset=str(fixed), formal_testset=str(formal), formal_remaining_testset=str(remaining), formal_baseline=str(formal_baseline),
        artifact_catalog=str(base.catalog_path),
    )
    data["model"].update(path=str(model), config=str(model_cfg))
    data["baseline"].update(source=str(source), secondary_source=str(secondary), centroids=str(centroids))
    config_path = project / "config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return Config(config_path, data)


def write_control(path: Path, artifact_type: str, run_id: str, **fields):
    value = control_envelope(artifact_type, run_id, "test", [], **fields)
    atomic_write_json(path, value)
    return value


def write_literature(path: Path):
    atomic_write_jsonl(path, [{"paper_id": "p1", "title": "Paper", "source_url": "https://example.test/p1", "provenance": "verified", "relevance": "core", "verification": "verified", "worker_id": "worker-00"}])


def architecture_candidates():
    lanes = (
        ("architecture", "kv_bit_allocation", "kv_bit_allocation_architecture"),
        ("architecture", "residual_token_retention", "residual_token_retention_architecture"),
        ("architecture", "representation_decoder", "representation_decoder_architecture"),
        ("stock_control", "stock_vecinfer_control", "stock_vecinfer_control"),
    )
    rows = [
        {
            "candidate_id": f"c{index + 1}", "candidate_kind": kind,
            "architecture_axis": axis, "canonical_mechanism_family": family,
            "mechanism_family": f"agent-label-{index}", "worker_id": f"worker-{index:02d}",
            "mechanism": f"architecture-{axis}", "causal_chain": "c", "storage_design": {},
            "falsification": "f", "novelty_risk": "low",
            "architecture_change": {
                "parameter_only": False, "state_before": "stock layout",
                "state_after": f"new {axis} runtime", "changed_runtime_components": [axis],
            },
        }
        for index, (kind, axis, family) in enumerate(lanes)
    ]
    rows[-1].pop("architecture_change")
    rows[-1]["control_invariants"] = {
        "selectable": False, "codebook_source": "baseline.centroids",
        "assignment_policy": "squared_euclidean", "candidate_artifacts_forbidden": True,
    }
    return rows
