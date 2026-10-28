from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from kvresearch.config import Config
from kvresearch.schema import ArtifactValidator, control_envelope
from kvresearch.stock_control import stock_codebook_manifest, validate_stock_control_file
from kvresearch.util import atomic_write_json, atomic_write_jsonl, path_ref
from kvresearch.vecinfer import VecInferAdapter
from kvresearch.pipeline import _cooldown_state
from kvresearch.initialize import initialize_run
from tests.helpers import architecture_candidates, test_config, write_control


class HostPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = test_config(self.root)
        data = copy.deepcopy(base.data)
        centroids = self.root / "stock-centroids"
        centroids.mkdir()
        storage = data["storage"]["vecinfer"]
        for layer in range(storage["layers"]):
            (centroids / f"key_cent_layer{layer}_{storage['key_subspaces']}_{storage['key_bits_per_code']}.pq.pt").write_bytes(f"k{layer}".encode())
            (centroids / f"val_cent_layer{layer}_{storage['value_subspaces']}_{storage['value_bits_per_code']}.pq.pt").write_bytes(f"v{layer}".encode())
        data["baseline"]["centroids"] = str(centroids)
        self.config = Config(base.path, data)

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_cannot_select_or_promote_stock_control(self):
        candidates = self.root / "candidates.jsonl"
        atomic_write_jsonl(candidates, architecture_candidates())
        bad = control_envelope(
            "selection", "run", "test", [path_ref(candidates)],
            selected_candidate_id="c4", ranking=["c4", "c1", "c2", "c3"], rejected=[], evidence=[],
        )
        path = self.root / "selection.json"
        atomic_write_json(path, bad)
        result = ArtifactValidator(self.config.catalog_path).validate_control(path, "selection")
        self.assertFalse(result.valid)
        self.assertTrue(any("architecture candidates" in error for error in result.errors))

    def test_stock_control_requires_real_root_hash_policy_and_no_candidate_loader(self):
        manifest = stock_codebook_manifest(self.config)
        path = self.root / "control.jsonl"
        row = {
            "arm_role": "stock_vecinfer_control",
            "stock_vecinfer_codebook_root": manifest["root"],
            "stock_vecinfer_codebook_manifest_sha256": manifest["sha256"],
            "stock_vecinfer_assignment_policy": "squared_euclidean",
        }
        atomic_write_jsonl(path, [row])
        self.assertEqual(validate_stock_control_file(self.config, path), [])
        row["candidate_codebook_root"] = "/tmp/candidate"
        atomic_write_jsonl(path, [row])
        self.assertTrue(any("candidate codebook" in error for error in validate_stock_control_file(self.config, path)))

    def test_adapter_injects_and_protects_stock_contract(self):
        manifest = stock_codebook_manifest(self.config)
        workspace = self.root / "workspace"
        workspace.mkdir()
        env = {
            "VECINFER_GENERATION_MAX_LENGTH": "2048",
            "VECINFER_MATCHED_STORAGE_LOGICAL_LENGTH": "16384",
        }
        plan = {"workspace": str(workspace), "execution": {"env": env}}
        actual = VecInferAdapter(self.config)._base_env(0, self.root, plan)
        self.assertEqual(actual["VECINFER_STOCK_CODEBOOK_ROOT"], manifest["root"])
        self.assertEqual(actual["VECINFER_STOCK_CODEBOOK_MANIFEST_SHA256"], manifest["sha256"])
        env["VECINFER_STOCK_CODEBOOK_ROOT"] = "/tmp/fake"
        with self.assertRaisesRegex(ValueError, "override host stock-control"):
            VecInferAdapter(self.config)._base_env(0, self.root, plan)

    def test_interleaved_family_and_agent_rename_do_not_reset_cooldown_streak(self):
        run = initialize_run(self.config, "canonical cooldown", "canonical-cooldown")

        def scientific_cycle(index, axis, family, label, score, streak=0):
            cycle = run / "cycles" / f"{index:03d}"
            atomic_write_jsonl(cycle / "candidates.jsonl", [{
                "candidate_id": "chosen", "architecture_axis": axis,
                "canonical_mechanism_family": family, "mechanism_family": label,
            }])
            write_control(cycle / "selection.json", "selection", run.name,
                          selected_candidate_id="chosen", ranking=[], rejected=[], evidence=[])
            write_control(cycle / "metrics.json", "metrics", run.name, status="complete", failures=[],
                          comparison_mode="development_candidate_ranking", longbench={"score": score},
                          gsm8k={"correct": 0}, math500={"correct": 0},
                          theoretical_storage_pass=True, actual_storage_pass=True)
            write_control(cycle / "experiment_diagnosis.json", "experiment_diagnosis", run.name,
                          measurement_valid=True)
            write_control(cycle / "decision.json", "decision", run.name, action="REFINE",
                          cooldown={"small_gain_streak": streak, "active_families": []})
            return cycle

        scientific_cycle(0, "kv_bit_allocation", "kv_bit_allocation_architecture", "alias-old", 10.0, 1)
        scientific_cycle(1, "representation_decoder", "representation_decoder_architecture", "interleaved", 20.0)
        scientific_cycle(2, "kv_bit_allocation", "kv_bit_allocation_architecture", "brand-new-name", 10.1)
        metrics = {
            "status": "complete", "failures": [], "comparison_mode": "development_candidate_ranking",
            "longbench": {"score": 10.1}, "gsm8k": {"correct": 0}, "math500": {"correct": 0},
            "theoretical_storage_pass": True, "actual_storage_pass": True,
        }
        state = _cooldown_state(self.config, run, "002", metrics, True)
        self.assertEqual(state["mechanism_family"], "kv_bit_allocation_architecture")
        self.assertEqual(state["small_gain_streak"], 2)
        self.assertTrue(state["triggered"])


if __name__ == "__main__":
    unittest.main()
