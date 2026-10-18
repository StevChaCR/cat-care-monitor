import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare.behavior_data import ClipLibrary
from catcare.behavior_runtime import BehaviorService
from catcare.config import DEFAULTS
from catcare.storage import Store


class BehaviorRuntimeTests(unittest.TestCase):
    def test_station_presence_alone_never_counts_and_lost_detections_pause_activity(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for simulated camera frames.")
        class Worker:
            results = {}
            def __init__(self, *args):
                pass
            def snapshot(self):
                return "ready", None, dict(self.results)
            def submit(self, *args):
                pass
            def stop(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "model.json"; manifest.write_text("{}")
            store = Store(root / "events.sqlite3")
            config = {**DEFAULTS, "zones": {"water": [.5, .5, .9, .9]}}
            with patch("catcare.behavior_runtime.ActivityWorker", Worker):
                service = BehaviorService(config, ClipLibrary(root / "clips"), store, manifest, root, "cpu")
            shape = (100, 100, 3)
            box = (55, 55, 85, 85)
            for timestamp in (1000, 1001, 1002):
                service.on_prediction([box], timestamp, shape)
                Worker.results["water"] = {"timestamp": timestamp, "probabilities": {"eating": .01, "drinking": .01, "other": .98}}
                service.poll(timestamp)
            self.assertEqual(store.activity_summary(1002)["water"]["sessions"], 0)
            for timestamp in (1003, 1004, 1005):
                service.on_prediction([box], timestamp, shape)
                Worker.results["water"] = {"timestamp": timestamp, "probabilities": {"eating": .01, "drinking": .98, "other": .01}}
                service.poll(timestamp)
            service.on_prediction([], 1006, shape); service.poll(1006)
            service.on_prediction([], 1020, shape); service.poll(1020)
            for timestamp in (1030, 1031, 1032):
                service.on_prediction([box], timestamp, shape)
                Worker.results["water"] = {"timestamp": timestamp, "probabilities": {"eating": .01, "drinking": .98, "other": .01}}
                service.poll(timestamp)
            summary = store.activity_summary(1032)["water"]
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(summary["observed_seconds"], 4)
            self.assertEqual(summary["incomplete"], 1)
            service.stop(); store.close()

    def test_opt_in_collection_records_real_cropped_sequences_without_action_labels(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("OpenCV and NumPy are needed for clip capture.")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = ClipLibrary(root / "clips")
            session = library.create_session("One recording period")
            store = Store(root / "events.sqlite3")
            config = {**DEFAULTS, "zones": {"food": [.05, .05, .4, .4], "water": [.6, .6, .95, .95]}}
            service = BehaviorService(config, library, store, None, root, "cpu")
            for index in range(20):
                frame = np.zeros((200, 300, 3), np.uint8)
                frame[:100, :, 0] = 100 + index
                frame[100:, :, 2] = 100 + index
                service.observe_frame(index, 1000 + index * .25, frame, session["id"])
            rows = library.clips()
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["station"] for row in rows}, {"food", "water"})
            self.assertTrue(all(row["label"] is None for row in rows))
            self.assertEqual(service.poll(1005)["phase"], "not_available")
            self.assertEqual(store.activity_summary(1005)["food"]["sessions"], 0)
            image = cv2.imread(str(library.frame_path(rows[0]["id"], 0)))
            self.assertLess(image.shape[1], 300)
            service.stop(); store.close()


if __name__ == "__main__":
    unittest.main()
