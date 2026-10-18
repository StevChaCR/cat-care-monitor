"""The learned head trains on sequences and loads numeric weights safely."""
import json
import tempfile
import unittest
from pathlib import Path

from catcare.behavior_model import FEATURES, classification_metrics


class ActivityHeadTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Install the application dependencies for activity-model checks.")
        torch.set_num_threads(2)
        self.np, self.torch = np, torch

    def test_training_evaluates_held_out_clips_and_round_trips_without_pickle(self):
        from catcare.behavior_model import fit_temporal_head, load_activity_head
        np, torch = self.np, self.torch
        rng = np.random.default_rng(42)
        features = rng.normal(size=(27, 16, FEATURES)).astype(np.float32)
        labels = [index % 3 for index in range(27)]
        for index, label in enumerate(labels):
            features[index, :, label * 100:label * 100 + 100] += 3
        splits = {"train": list(range(9)), "val": list(range(9, 18)), "test": list(range(18, 27))}
        events = []
        with tempfile.TemporaryDirectory() as folder:
            manifest, metrics = fit_temporal_head(features, labels, splits, folder, 5, "cpu",
                lambda phase, **values: events.append((phase, values)))
            self.assertEqual(metrics["clips"], 9)
            self.assertEqual(metrics["evaluation_split"], "test")
            self.assertEqual(sum(row["examples"] for row in metrics["per_class"].values()), 9)
            self.assertEqual(events[-1][0], "completed")
            head, metadata = load_activity_head(manifest)
            with torch.inference_mode():
                scores = head(torch.from_numpy(features[:1]).transpose(1, 2)).softmax(1)
            self.assertEqual(tuple(scores.shape), (1, 3))
            self.assertTrue(torch.isfinite(scores).all())
            weights = Path(folder) / "weights.npz"
            with weights.open("ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_activity_head(manifest)

    def test_classification_report_distinguishes_false_positives_and_missed_actions(self):
        metrics = classification_metrics([0, 0, 1, 1, 2, 2], [0, 2, 1, 2, 0, 2])
        self.assertEqual(metrics["per_class"]["eating"]["precision"], .5)
        self.assertEqual(metrics["per_class"]["drinking"]["precision"], 1)
        self.assertEqual(metrics["per_class"]["drinking"]["recall"], .5)


if __name__ == "__main__":
    unittest.main()
