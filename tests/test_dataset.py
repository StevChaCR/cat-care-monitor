import json
import tempfile
import unittest
from pathlib import Path
from catcare.dataset import prepare_dataset, read_labels


class DatasetTests(unittest.TestCase):
    def create_sessions(self, root):
        for index in range(6):
            session = root / f"session_{index}"
            (session / "images").mkdir(parents=True)
            (session / "labels").mkdir()
            (session / "images" / "positive.jpg").write_bytes(f"test-positive-{index}".encode())
            (session / "labels" / "positive.txt").write_text("0 0.5 0.5 0.4 0.6\n")
            (session / "images" / "negative.jpg").write_bytes(f"test-negative-{index}".encode())
            (session / "labels" / "negative.txt").write_text("")

    def test_session_split_and_negative_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_sessions(root / "captures")
            report = prepare_dataset(root / "captures", root / "dataset")
            groups = [set(report["splits"][name]["sessions"]) for name in ("train", "val", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertEqual(sum(report["splits"][name]["images"] for name in report["splits"]), 12)
            self.assertIn("0: cat", (root / "dataset" / "data.yaml").read_text())
            copied = list((root / "dataset" / "labels").rglob("*.txt"))
            self.assertEqual(sum(not path.read_text() for path in copied), 6)
            with self.assertRaises(ValueError):
                prepare_dataset(root / "captures", root / "dataset")

    def test_exact_duplicate_sessions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_sessions(root / "captures")
            a = root / "captures" / "session_0" / "images" / "positive.jpg"
            b = root / "captures" / "session_1" / "images" / "positive.jpg"
            b.write_bytes(a.read_bytes())
            with self.assertRaises(ValueError):
                prepare_dataset(root / "captures", root / "dataset")

    def test_unreviewed_images_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_sessions(root / "captures")
            (root / "captures" / "session_0" / "labels" / "positive.txt").unlink()
            with self.assertRaises(ValueError):
                prepare_dataset(root / "captures", root / "dataset")

    def test_out_of_bounds_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.95 0.5 0.4 0.4\n")
            with self.assertRaises(ValueError):
                read_labels(path)
