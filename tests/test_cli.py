import subprocess
import sys
import unittest
from pathlib import Path


class CommandLineTests(unittest.TestCase):
    def test_accidental_url_argument_is_not_echoed(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root / "app.py"),
                                 "rtsp://unit:fake@192.168.1.20/live"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("unit:fake", result.stderr)
        self.assertNotIn("192.168.1.20", result.stderr)
        self.assertIn("desktop dialog", result.stderr)
