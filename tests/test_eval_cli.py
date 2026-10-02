import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvalCLITests(unittest.TestCase):
    def test_isolated_evaluator_records_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream, predictions = root / "official", root / "predictions"
            upstream.mkdir()
            predictions.mkdir()
            (upstream / "pdf_validation.py").write_text(
                "import sys\nfrom pathlib import Path\n"
                "assert Path.cwd() != Path(__file__).parent\n"
                "Path('result').mkdir()\nraise SystemExit(7)\n"
            )
            gt = root / "annotations.json"
            gt.write_text("[]")
            config = root / "config.yaml"
            config.write_text("benchmark:\n  root: unused\nbaselines: []\n")
            script = Path(__file__).resolve().parents[1] / "scripts/eval_omnidocbench.py"
            args = [
                sys.executable,
                str(script),
                "--config",
                str(config),
                "--official-repo",
                str(upstream),
                "--pred-dir",
                str(predictions),
                "--gt",
                str(gt),
                "--output-root",
                str(root / "runs"),
                "--run-id",
                "fixture",
            ]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertFalse((upstream / "result").exists())
            metadata = json.loads((root / "runs/fixture/run.json").read_text())
            self.assertEqual(metadata["returncode"], 7)
            self.assertTrue((root / "runs/fixture/result").is_dir())
            repeated = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
