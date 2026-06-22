"""Phase 5: scaffold tests for eval/run_grid.py (non-model parts: instance
metadata, the grid loop via DryRunSolver, monotonicity summary, CSV writing).
The real RL path needs a checkpoint and is validated separately.

Run: uv run python -m unittest tests.test_eval_grid -v
"""
import csv
import os
import tempfile
import unittest

OOD = "data/ood"


@unittest.skipUnless(os.path.isdir(OOD), "data/ood not present")
class TestEvalGrid(unittest.TestCase):
    def setUp(self):
        from eval.run_grid import iter_files
        self.files = iter_files(OOD, limit=4)
        self.assertTrue(self.files, "no .npz under data/ood")

    def test_read_meta(self):
        from eval.run_grid import read_meta
        m = read_meta(self.files[0])
        for k in ("file", "topology", "n_req", "density"):
            self.assertIn(k, m)
        self.assertGreater(m["n_req"], 0)

    def test_dry_grid_rows_and_monotone(self):
        from eval.run_grid import DryRunSolver, run_grid
        Ms = [2, 3, 5, 7]
        rows = run_grid(DryRunSolver(), self.files, Ms, ["P"], num_sample=10)
        self.assertEqual(len(rows), len(self.files) * len(Ms))
        # synthetic T decreases with M -> per file, T1 strictly non-increasing
        by_file = {}
        for r in rows:
            by_file.setdefault(r["file"], {})[r["M"]] = r["T1"]
        for fM in by_file.values():
            seq = [fM[m] for m in Ms]
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(seq, seq[1:])))

    def test_write_csv(self):
        from eval.run_grid import DryRunSolver, run_grid, write_csv
        rows = run_grid(DryRunSolver(), self.files[:2], [3, 5], ["P"], num_sample=10)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.csv")
            write_csv(rows, out)
            with open(out) as fh:
                got = list(csv.DictReader(fh))
            self.assertEqual(len(got), len(rows))
            self.assertIn("T1", got[0])


if __name__ == "__main__":
    unittest.main()
