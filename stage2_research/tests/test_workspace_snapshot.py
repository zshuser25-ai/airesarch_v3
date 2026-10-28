from __future__ import annotations

import unittest

from kvresearch.workspace import _snapshot_status_paths


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_expands_file_status_and_excludes_runtime_noise(self):
        status = "\n".join([
            " M scripts/modeldb/main_pq.py",
            "?? tests/test_candidate.py",
            "?? tests/__pycache__/test_candidate.cpython-311.pyc",
            "?? scripts/modeldb/__pycache__/main_pq.cpython-311.pyc",
            "?? centroids",
            "?? models",
            "?? scales",
            "?? scripts/modeldb/results.jsonl",
            "?? candidate_artifacts/table.pt",
        ])
        self.assertEqual(
            _snapshot_status_paths(status),
            [
                "candidate_artifacts/table.pt",
                "scripts/modeldb/main_pq.py",
                "tests/test_candidate.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
