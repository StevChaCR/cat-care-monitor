"""Shared detector selection, immutable evaluation groups, and release integrity."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare import config
from catcare.dataset import prepare_dataset
from catcare.models import bundled_detector, resolve_detector
from catcare.release import detector_report
from catcare.web_runtime import WebRuntime
from catcare.web_training import TrainingJob


def make_bundle(root):
    folder = root / "assets" / "detector"
    folder.mkdir(parents=True)
    weights = folder / "model.pt"
    weights.write_bytes(b"integrity-fixture-not-a-neural-network")
    manifest = {"schema_version": 1, "file": "model.pt", "classes": ["cat"], "task": "detect",
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (folder / "model.json").write_text(json.dumps(manifest))
    return weights


class SharedDetectorTests(unittest.TestCase):
    def test_new_installations_use_included_weights_and_existing_custom_weights_remain_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = make_bundle(root)
            with patch.object(config, "ROOT", root), patch.object(config, "LOCAL", root / "local"):
                self.assertEqual(config.load_settings()["model"], "bundled")
                self.assertEqual(resolve_detector("bundled"), weights)
                private = root / "private" / "best.pt"
                private.parent.mkdir(); private.write_bytes(b"existing-private-detector")
                config.save_settings({**copy.deepcopy(config.DEFAULTS), "model": str(private)})
                self.assertEqual(config.load_settings()["model"], str(private))

    def test_bundled_resources_reject_tampering_and_manifest_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = make_bundle(root)
            self.assertEqual(bundled_detector(weights.parent), weights)
            weights.write_bytes(b"changed-after-release")
            with self.assertRaisesRegex(ValueError, "checksum"):
                bundled_detector(weights.parent)
            manifest = weights.parent / "model.json"
            row = json.loads(manifest.read_text()); row["file"] = "../../private.pt"
            manifest.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "invalid"):
                bundled_detector(weights.parent)

    def test_camera_settings_do_not_reset_the_current_detector_to_a_generic_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(config, "ROOT", root), patch.object(config, "LOCAL", root / "local"), \
                    patch("catcare.web_runtime.LOCAL", root / "local"):
                runtime = WebRuntime()
                owned = root / "best.pt"; owned.write_bytes(b"selected")
                runtime.config["model"] = str(owned)
                runtime.settings({"image_size": 960})
                self.assertEqual(runtime.config["model"], str(owned))
                with self.assertRaisesRegex(ValueError, "current detector"):
                    runtime.settings({"model": "yolo26n.pt"})
                self.assertEqual(runtime.config["model"], str(owned))

    def test_release_metrics_are_tied_to_the_evaluated_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights" / "best.pt"
            weights.parent.mkdir(); weights.write_bytes(b"evaluated")
            report = {"evaluation_split": "test", "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                      "precision": .9, "recall": .8, "mAP50": .85, "mAP50_95": .7}
            (root / "held_out_metrics.json").write_text(json.dumps(report))
            self.assertEqual(detector_report(weights)[1]["recall"], .8)
            weights.write_bytes(b"different-weights")
            with self.assertRaisesRegex(ValueError, "changed after evaluation"):
                detector_report(weights)


class FrozenSplitTests(unittest.TestCase):
    def sessions(self, root, names):
        for name in names:
            folder = root / name
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir()
            (folder / "images" / "cat.jpg").write_bytes(name.encode())
            (folder / "labels" / "cat.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    def test_new_periods_do_not_move_previous_holdouts_into_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sessions(root / "sessions", [f"period-{index}" for index in range(6)])
            first = prepare_dataset(root / "sessions", root / "first", split_file=root / "split.json")
            self.sessions(root / "sessions", ["new-period-a", "new-period-b"])
            later = prepare_dataset(root / "sessions", root / "later", split_file=root / "split.json")
            for split in ("val", "test"):
                self.assertEqual(first["splits"][split]["sessions"], later["splits"][split]["sessions"])
            self.assertTrue({"new-period-a", "new-period-b"} <= set(later["splits"]["train"]["sessions"]))

    def test_existing_checkpoint_split_is_inherited_when_continuing_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sessions(root / "sessions", ["old-a", "old-b", "old-c", "old-d"])
            original = prepare_dataset(root / "sessions", root / "old", seed=7)
            inherited = prepare_dataset(root / "sessions", root / "new", split_file=root / "split.json",
                                        inherit_manifest=root / "old" / "split_manifest.json")
            self.assertEqual(inherited["seed"], 7)
            for split in ("train", "val", "test"):
                self.assertEqual(set(original["splits"][split]["sessions"]), set(inherited["splits"][split]["sessions"]))

    def test_training_process_receives_the_selected_checkpoint_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sessions(root / "sessions", ["a", "b", "c"])
            weights = root / "best.pt"; weights.write_bytes(b"current-detector")
            with patch("catcare.web_training.subprocess.Popen") as process:
                job = TrainingJob()
                job.start(root / "profile", root / "sessions", str(weights), 20, 960, "0")
                command = process.call_args.args[0]
                self.assertEqual(command[command.index("--model") + 1], str(weights))
                self.assertEqual(weights.read_bytes(), b"current-detector")
                self.assertTrue((root / "profile" / "detector_split.json").is_file())


if __name__ == "__main__":
    unittest.main()
