import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_edr import Observation
from ocr_edr.loop import digest
from ocr_edr.render import CommandRenderer


class RenderTests(unittest.TestCase):
    def test_each_render_has_its_own_output_and_current_markup_hash(self):
        script = "import base64,sys; from pathlib import Path; Path(sys.argv[2]).write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6jGAAAAAASUVORK5CYII='))"
        with tempfile.TemporaryDirectory() as tmp:
            renderer = CommandRenderer(
                [sys.executable, "-c", script, "{input}", "{output}"], Path(tmp)
            )
            a = renderer.render(Observation("a", "formula", "source.png", "x^2"))
            b = renderer.render(Observation("a", "formula", "source.png", "x^3"))
            self.assertNotEqual(a.path, b.path)
            self.assertEqual(a.prediction_sha256, digest("x^2"))
            self.assertEqual(b.prediction_sha256, digest("x^3"))

    def test_renderer_error_propagates_for_loop_to_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            renderer = CommandRenderer(
                [sys.executable, "-c", "raise SystemExit(1)", "{input}", "{output}"], Path(tmp)
            )
            with self.assertRaises(subprocess.CalledProcessError):
                renderer.render(Observation("a", "table", "source.png", "<table/>"))

    def test_missing_placeholders_rejected(self):
        with self.assertRaises(ValueError):
            CommandRenderer(["renderer"], Path("unused"))


if __name__ == "__main__":
    unittest.main()
