"""Tests for train.py's checkpoint guard: a new run must not replace a checkpoint that is on disk.
Every case runs train.py in a temporary folder. The guard acts before any data is read, so no data,
no GPU and no training is needed; a run that gets past the guard fails right after it, on the missing data.

    .venv\\Scripts\\python.exe -m unittest test_train
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).parent


class TestCheckpointGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        for name in ("train.py", "model.py"):
            shutil.copy(REPO / name, self.folder / name)

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, *args):
        return subprocess.run([sys.executable, "train.py", *args], cwd=self.folder,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def trained(self, name):
        """A stand-in for a trained checkpoint: the guard only looks at whether the file exists."""
        path = self.folder / name
        path.write_bytes(b"a trained model")
        return path

    def assert_stopped_by_guard(self, result):
        self.assertEqual(result.returncode, 2)
        self.assertIn("is taken", result.stderr)

    def assert_passed_guard(self, result):
        self.assertNotIn("is taken", result.stderr)
        self.assertIn("meta.json", result.stderr)          # it went on and failed on the missing data folder

    def test_stops_when_the_best_checkpoint_is_on_disk(self):
        ckpt = self.trained("x.pt")
        self.assert_stopped_by_guard(self.train("--name", "x"))
        self.assertEqual(ckpt.read_bytes(), b"a trained model")

    def test_stops_when_only_the_last_checkpoint_is_on_disk(self):
        ckpt = self.trained("x_last.pt")
        self.assert_stopped_by_guard(self.train("--name", "x"))
        self.assertEqual(ckpt.read_bytes(), b"a trained model")

    def test_a_free_name_passes(self):
        self.trained("x.pt")
        self.assert_passed_guard(self.train("--name", "y"))

    def test_overwrite_passes(self):
        self.trained("x.pt")
        self.assert_passed_guard(self.train("--name", "x", "--overwrite"))

    def test_resume_without_a_checkpoint_says_what_is_missing(self):
        result = self.train("--name", "x", "--resume")
        self.assertEqual(result.returncode, 2)
        self.assertIn("x_last.pt", result.stderr)
        self.assertIn("not on disk", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_resume_and_overwrite_together_are_refused(self):
        self.trained("x_last.pt")
        result = self.train("--name", "x", "--resume", "--overwrite")
        self.assertEqual(result.returncode, 2)
        self.assertIn("pass only one of them", result.stderr)


if __name__ == "__main__":
    unittest.main()
