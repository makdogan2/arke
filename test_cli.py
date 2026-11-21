"""Tests for the command lines of the scripts.

Every script that takes arguments must answer --help, and arguments that cannot work must be refused
with a message before anything is loaded. The scripts run as copies in an empty temporary folder:
there are no data files and no checkpoints next to them, so whatever a script does, it cannot touch
the real ones. (A script without arguments, such as tokenizer_bpe.py, starts working as soon as it
runs. Never try --help on those in the repository itself.)

    .venv\\Scripts\\python.exe -m unittest test_cli
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS = ["train.py", "sample.py", "sample_stats.py", "quantize.py", "blind_pairs.py", "data_prep.py",
           "data_download.py", "plot_scaling.py", "plot_seq_warmup.py"]     # the scripts that take arguments
IMPORTED = ["model.py", "textclean.py"]                                      # what they import from the repo

# script, arguments, a piece of the message the script must give
REFUSED = [
    ("sample.py", ["--temperature", "0"], "--temperature must be above 0"),
    ("sample.py", ["--top_p", "1.5"], "--top_p must be above 0 and at most 1"),
    ("sample.py", ["--repetition_penalty", "0"], "--repetition_penalty must be above 0"),
    ("sample.py", ["--top_k", "-1"], "--top_k must be 0 (off) or a positive number"),
    ("sample.py", ["--tokens", "0"], "--tokens must be at least 1"),
    ("sample_stats.py", ["--temperature", "0"], "--temperature must be above 0"),
    ("sample_stats.py", ["--top_k", "-3"], "--top_k must be 0 (off) or a positive number"),
    ("quantize.py", ["--sample", "--window", "128"], "--sample measures no loss"),
    ("quantize.py", ["--load", "x.pt", "--packed"], "--load takes everything from the file"),
    ("quantize.py", ["--packed"], "--packed only does 4-bit asym"),
    ("quantize.py", ["--save", "x.pt"], "--save needs --packed"),
    ("data_prep.py", ["--smoke", "--out", "v2"], "--smoke writes to data/prep_smoke"),
]


class TestCommandLines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        for name in SCRIPTS + IMPORTED + ["count_junk.py"]:
            shutil.copy(ROOT / name, folder / name)

        def run(call):
            script, args = call
            return subprocess.run([sys.executable, script, *args], cwd=folder, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=300)

        calls = [(s, ["--help"]) for s in SCRIPTS] + [(s, a) for s, a, _ in REFUSED] + [("count_junk.py", [])]
        with ThreadPoolExecutor(max_workers=8) as pool:      # each call starts a Python and imports torch: side by side
            results = list(pool.map(run, calls))
        cls.results = {(script, tuple(args)): r for (script, args), r in zip(calls, results)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_script_answers_help(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                r = self.results[(script, ("--help",))]
                self.assertEqual(r.returncode, 0, r.stderr[-400:])
                self.assertIn("usage:", r.stdout)

    def test_arguments_that_cannot_work_are_refused_with_a_message(self):
        for script, args, message in REFUSED:
            with self.subTest(script=script, args=" ".join(args)):
                r = self.results[(script, tuple(args))]
                self.assertEqual(r.returncode, 2, r.stderr[-400:])     # argparse's exit code for a refused argument
                self.assertIn(message, r.stderr)
                self.assertNotIn("Traceback", r.stderr)                 # refused by the script, not crashed

    def test_count_junk_gets_as_far_as_looking_for_the_corpus(self):
        r = self.results[("count_junk.py", ())]      # it takes no arguments and reads data/mix.txt, which is not there
        self.assertEqual(r.returncode, 1, r.stderr[-400:])
        self.assertIn("Run data_download.py first", r.stdout)
        self.assertNotIn("Traceback", r.stderr)      # so its imports work


if __name__ == "__main__":
    unittest.main()
