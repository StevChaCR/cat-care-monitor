"""Station selection and recoverable cleanup of private behavior sequences."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from catcare.behavior_data import ClipLibrary
from catcare.behavior_runtime import BehaviorService
from catcare.config import DEFAULTS
from catcare.storage import Store
from catcare.web_runtime import WebRuntime


class ClipCleanupTests(unittest.TestCase):
    def setup_library(self, folder):
        library = ClipLibrary(Path(folder) / "sessions")
        session = library.create_session("Real recording period")
        clips = []
        for index in range(3):
            row = library.add_clip(session["id"], "water" if index % 2 else "food",
                                   tuple(f"clip-{index}-frame-{f}".encode() for f in range(16)),
                                   [1000 + index * 20 + f * .25 for f in range(16)], [0, 0, 100, 100])
            library.label_clip(row["id"], "exclude" if index == 0 else "other")
            clips.append(row)
        return library, clips

    def test_removal_is_recoverable_and_retains_frames_labels_and_recording_period(self):
        with tempfile.TemporaryDirectory() as folder:
            library, clips = self.setup_library(folder)
            key = clips[0]["id"]
            before = library.frame_path(key, 0).read_bytes()
            self.assertEqual(library.clips()[0].get("frames"), 16)
            self.assertEqual(len(library.clips()), 3)  # Cannot tell alone retains files.
            self.assertEqual(library.move_to_trash([key]), 1)
            self.assertEqual(len(library.clips()), 2)
            with self.assertRaises(ValueError):
                library.frame_path(key, 0)
            self.assertEqual(library.frame_path(key, 0, trashed=True).read_bytes(), before)
            trashed = library.trashed_clips()[0]
            self.assertEqual(trashed["label"], "exclude")
            self.assertEqual(trashed["session_name"], "Real recording period")
            library.restore_clip(key)
            self.assertEqual(library.trashed_clips(), [])
            self.assertEqual(library.frame_path(key, 0).read_bytes(), before)
            row = json.loads((library.clip_path(key) / "clip.json").read_text())
            self.assertEqual(row["session_id"], clips[0]["session_id"])
            self.assertEqual(row["label"], "exclude")

    def test_purge_erases_only_requested_trash_and_preserves_active_clips_and_models(self):
        with tempfile.TemporaryDirectory() as folder:
            library, clips = self.setup_library(folder)
            weights = Path(folder) / "best.pt"
            weights.write_bytes(b"previous trained checkpoint")
            library.move_to_trash([row["id"] for row in clips[:2]])
            self.assertEqual(library.purge_clips([clips[0]["id"]]), 1)
            self.assertEqual([row["id"] for row in library.trashed_clips()], [clips[1]["id"]])
            self.assertTrue(library.frame_path(clips[2]["id"], 0).is_file())
            self.assertEqual(weights.read_bytes(), b"previous trained checkpoint")
            with self.assertRaises(ValueError):
                library.purge_clips([clips[2]["id"]])  # Active clips cannot be purged.

    def test_invalid_batch_or_foreign_reference_does_not_move_or_erase_valid_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            library, clips = self.setup_library(folder)
            missing = "a" * 32 + "_" + "b" * 32
            for keys in ([], [clips[0]["id"], missing], ["../private"], [clips[0]["id"]] * 2):
                with self.assertRaises(ValueError):
                    library.move_to_trash(keys)
            self.assertEqual(len(library.clips()), 3)
            library.move_to_trash([clips[0]["id"]])
            with self.assertRaises(ValueError):
                library.purge_clips([clips[0]["id"], missing])
            self.assertTrue(library.frame_path(clips[0]["id"], 0, trashed=True).is_file())
            foreign = ClipLibrary(Path(folder) / "other" / "sessions")
            with self.assertRaises(ValueError):
                foreign.restore_clip(clips[0]["id"])

    def test_trash_symlink_cannot_redirect_cleanup_outside_the_library(self):
        with tempfile.TemporaryDirectory() as folder:
            library, clips = self.setup_library(folder)
            external = Path(folder) / "unrelated"
            external.mkdir()
            try:
                (library.root / "trash").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("Creating directory symlinks is unavailable.")
            with self.assertRaises(ValueError):
                library.move_to_trash([clips[0]["id"]])
            self.assertTrue(library.frame_path(clips[0]["id"], 0).is_file())


class BehaviorCollectionControlsTests(unittest.TestCase):
    def test_area_selection_limits_recording_but_other_station_recognition_continues(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("OpenCV and NumPy are needed for simulated station capture.")
        with tempfile.TemporaryDirectory() as folder, ExitStack() as cleanup:
            root = Path(folder)
            library = ClipLibrary(root / "sessions")
            session = library.create_session("One real period")
            store = Store(root / "events.sqlite3")
            cleanup.callback(store.close)
            config = {**DEFAULTS, "zones": {"food": [.05, .05, .4, .4], "water": [.6, .6, .95, .95]}}
            service = BehaviorService(config, library, store, None, root, "cpu")
            service.worker = MagicMock()
            cleanup.callback(service.stop)
            frame = np.zeros((200, 300, 3), np.uint8)
            for index in range(16):
                stamp = 1000 + index * .25
                service.on_prediction([(20, 20, 90, 70)], stamp, frame.shape)
                service.observe_frame(index, stamp, frame, session["id"], ("water",), 1)
            self.assertEqual([row["station"] for row in library.clips()], ["water"])
            self.assertEqual(service.worker.submit.call_args.args[0], "food")

    def test_only_selected_station_is_saved_and_switching_area_resets_sequence(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest("OpenCV and NumPy are needed for simulated station capture.")
        with tempfile.TemporaryDirectory() as folder, ExitStack() as cleanup:
            root = Path(folder)
            library = ClipLibrary(root / "sessions")
            session = library.create_session("One recording period")
            store = Store(root / "events.sqlite3")
            cleanup.callback(store.close)
            config = {**DEFAULTS, "zones": {"food": [.05, .05, .4, .4], "water": [.6, .6, .95, .95]}}
            service = BehaviorService(config, library, store, None, root, "cpu")
            cleanup.callback(service.stop)
            frame = np.zeros((200, 300, 3), np.uint8)
            def observe(index, station, generation):
                frame[:, :, 0] = index
                service.observe_frame(index, 1000 + index * .25, frame, session["id"], (station,), generation)
            for index in range(16):
                observe(index, "water", 1)
            rows = library.clips()
            self.assertEqual([row["station"] for row in rows], ["water"])
            self.assertIsNone(rows[0]["label"])
            self.assertEqual(service.poll(1004)["collection_stations"], ["water"])
            for index in range(16, 31):
                observe(index, "food", 2)
            self.assertEqual(len(library.clips()), 1)  # Not enough frames in the new area yet.
            observe(31, "food", 2)
            self.assertEqual({row["station"] for row in library.clips()}, {"food", "water"})
            food = next(row for row in library.clips() if row["station"] == "food")
            self.assertEqual(food["timestamps"][0], 1004)
            self.assertTrue(all(row["label"] is None for row in library.clips()))
            # A fast stop/start in the same session also starts a fresh sequence.
            for index in range(32, 47):
                observe(index, "food", 3)
            self.assertEqual(len(library.clips()), 2)
            observe(47, "food", 3)
            self.assertEqual(len(library.clips()), 3)

    def test_runtime_validates_area_and_does_not_mutate_clips_during_training(self):
        with tempfile.TemporaryDirectory() as folder, patch("catcare.config.LOCAL", Path(folder)), \
                patch("catcare.web_runtime.LOCAL", Path(folder)):
            runtime = WebRuntime()
            runtime.config["zones"] = {"water": [.5, .5, .9, .9]}
            session = runtime.behavior_library.create_session("Evening")
            with patch.object(runtime, "active", return_value=True):
                for station in ("food", "litter_box", ["water"]):
                    with self.assertRaises(ValueError):
                        runtime.collect_behavior(True, session["id"], station)
                runtime.collect_behavior(True, session["id"], "water")
                self.assertEqual(runtime.behavior_capture_stations, ("water",))
                generation = runtime.behavior_capture_generation
                runtime.collect_behavior(False)
                runtime.collect_behavior(True, session["id"], "both")
                self.assertGreater(runtime.behavior_capture_generation, generation)
                self.assertEqual(runtime.behavior_capture_stations, ("water",))
            with patch.object(runtime, "training_busy", return_value=True):
                for operation in ("trash", "restore", "purge"):
                    with self.assertRaisesRegex(ValueError, "Stop training"):
                        runtime.manage_behavior_clips(operation, [])
                with self.assertRaisesRegex(ValueError, "Stop training"):
                    runtime.label_behavior_clip("anything", "other")


if __name__ == "__main__":
    unittest.main()
