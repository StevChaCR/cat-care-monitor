"""Activity evidence, observation gaps, private clips, and whole-session splits."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare.behavior import ActivityCounter, ActivitySession, activity_evidence, CLIP_FRAMES
from catcare.behavior_data import ClipLibrary, prepare_behavior_dataset
from catcare.storage import Store


class ActivityCounterTests(unittest.TestCase):
    def start(self):
        counter = ActivityCounter()
        counter.observe(False, 0)
        counter.observe(False, 6)
        for stamp in (10, 11, 12):
            events = counter.observe(True, stamp)
        self.assertEqual(events[0][0], "start")
        return counter

    def test_resting_near_station_never_counts_as_eating(self):
        counter = ActivityCounter()
        for timestamp in range(100):
            counter.observe(False, timestamp)
        self.assertIsNone(counter.session)

    def test_brief_action_score_does_not_create_a_session(self):
        counter = ActivityCounter()
        counter.observe(True, 0)
        counter.observe(True, 1)
        counter.observe(False, 2)
        self.assertIsNone(counter.session)

    def test_occlusion_retains_one_session_and_does_not_fill_unknown_time(self):
        counter = self.start()
        session = counter.session
        counter.observe(None, 13)
        counter.observe(None, 40)
        self.assertEqual(counter.observe(True, 50)[0][0], "update")
        counter.observe(True, 51)
        self.assertIs(counter.session, session)
        self.assertEqual(session.intervals, [[10, 12], [50, 51]])
        self.assertTrue(session.uncertain)

    def test_long_unobserved_gap_cannot_bridge_duration(self):
        counter = self.start()
        counter.observe(True, 100)
        self.assertEqual(counter.session.intervals, [[10, 12]])
        self.assertTrue(counter.session.uncertain)

    def test_reliable_exit_closes_and_later_activity_is_a_new_session(self):
        counter = self.start()
        first = counter.session
        counter.observe(False, 13)
        self.assertEqual(counter.observe(False, 19)[0][0], "end")
        self.assertEqual(first.status, "completed")
        for stamp in (20, 21, 22):
            result = counter.observe(True, stamp)
        self.assertEqual(result[0][0], "start")
        self.assertIsNot(counter.session, first)

    def test_duplicate_or_out_of_order_predictions_do_not_advance_time(self):
        counter = self.start()
        for timestamp in (12, 12, 11, 10):
            self.assertEqual(counter.observe(True, timestamp), [])
        self.assertEqual(counter.session.intervals, [[10, 12]])

    def test_interruption_and_resume_preserve_identity_without_counting_offline_time(self):
        first = self.start()
        interrupted = first.interrupt()
        interrupted.row_id = 7
        resumed = ActivityCounter(resume=interrupted)
        resumed.observe(True, 100)
        resumed.observe(True, 101)
        self.assertEqual(resumed.session.row_id, 7)
        self.assertEqual(resumed.session.intervals, [[10, 12], [100, 101]])

    def test_wrong_station_low_score_and_ambiguous_scores_are_unknown(self):
        self.assertIsNone(activity_evidence("food", {"eating": .02, "drinking": .96, "other": .02}))
        self.assertIsNone(activity_evidence("water", {"eating": .1, "drinking": .6, "other": .3}))
        self.assertIs(activity_evidence("water", {"eating": .01, "drinking": .98, "other": .01}), True)
        self.assertIs(activity_evidence("food", {"eating": .01, "drinking": .01, "other": .98}), False)
        self.assertIsNone(activity_evidence("food", {"eating": float("nan"), "drinking": 0, "other": 0}))


class ActivityStorageTests(unittest.TestCase):
    def test_packaged_app_data_survives_outside_temporary_executable_resources(self):
        from catcare.config import data_directory, ROOT
        with patch("sys.frozen", False, create=True):
            self.assertEqual(data_directory(), ROOT / "local")
        with tempfile.TemporaryDirectory() as temporary, patch("sys.frozen", True, create=True), \
                patch("sys.platform", "win32"), patch.dict("os.environ", {"LOCALAPPDATA": temporary}):
            self.assertEqual(data_directory(), Path(temporary) / "WatchMyCat")

    def test_rolling_summary_clips_real_intervals_and_resume_uses_same_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.sqlite3"
            store = Store(path)
            session = ActivitySession(0, 110, [[0, 10], [100, 110]], uncertain=True)
            store.write_activity("water", session, "model-a")
            original_id = session.row_id
            session.status = "interrupted"
            store.write_activity("water", session, "model-a")
            self.assertEqual(store.activity_summary(110, window_seconds=15)["water"]["observed_seconds"], 10)
            self.assertEqual(store.activity_summary(110, window_seconds=200)["water"]["sessions"], 1)
            self.assertIsNone(store.resume_activity("water", "different-model"))
            resumed = store.resume_activity("water", "model-a")
            self.assertEqual(resumed.row_id, original_id)
            store.write_activity("water", resumed, "model-a")
            self.assertEqual(store.db.execute("SELECT count(*) FROM activities").fetchone()[0], 1)
            store.close()


class ClipDataTests(unittest.TestCase):
    def create(self, root):
        library = ClipLibrary(root)
        rows = []
        for group in range(3):
            session = library.create_session(f"Different day {group}")
            for index, label in enumerate(["eating"] * 3 + ["drinking"] * 3 + ["other"] * 4):
                station = "food" if label == "eating" or (label == "other" and index % 2 == 0) else "water"
                frames = tuple(f"frame-{group}-{index}-{f}".encode() for f in range(CLIP_FRAMES))
                stamps = [group * 86400 + index * 20 + .25 * f for f in range(CLIP_FRAMES)]
                row = library.add_clip(session["id"], station, frames, stamps, [0, 0, 640, 480])
                library.label_clip(row["id"], label)
                rows.append(row)
        return library, rows

    def test_activity_labels_are_manual_and_invalid_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            library = ClipLibrary(folder)
            session = library.create_session("Evening")
            row = library.add_clip(session["id"], "water", tuple(f"frame-{i}".encode() for i in range(16)),
                                   [10 + i * .25 for i in range(16)], [0, 0, 100, 100])
            self.assertIsNone(library.clips()[0]["label"])
            for key in ("../settings", "a" * 32 + "_../private"):
                with self.assertRaises(ValueError):
                    library.frame_path(key, 0)
            with self.assertRaises(ValueError):
                library.frame_path(row["id"], -1)
            library.label_clip(row["id"], "exclude")
            self.assertEqual(library.clips()[0]["label"], "exclude")

    def test_sparse_or_discontinuous_camera_sequences_are_not_training_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            library = ClipLibrary(folder)
            session = library.create_session("One period")
            with self.assertRaises(ValueError):
                library.add_clip(session["id"], "food", (b"frame",) * 16,
                                 list(range(16)), [0, 0, 100, 100])

    def test_split_keeps_whole_periods_and_requires_hard_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library, _ = self.create(root / "source")
            result = prepare_behavior_dataset(library.root, root / "dataset.json")
            groups = [set(row["sessions"]) for row in result["splits"].values()]
            self.assertTrue(groups[0].isdisjoint(groups[1]))
            self.assertTrue(groups[1].isdisjoint(groups[2]))
            self.assertEqual(sum(len(row["clips"]) for row in result["splits"].values()), 30)
            for clip in library.clips():
                if clip["station"] == "water" and clip["label"] == "other":
                    library.label_clip(clip["id"], "exclude")
            with self.assertRaisesRegex(ValueError, "Other examples at both"):
                prepare_behavior_dataset(library.root, root / "another.json")

    def test_unreviewed_clip_and_fake_overlapping_periods_block_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library, rows = self.create(root / "source")
            path = library.clip_path(rows[0]["id"]) / "clip.json"
            row = json.loads(path.read_text())
            row["label"] = None
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Review every"):
                prepare_behavior_dataset(library.root, root / "unreviewed.json")
            row["label"] = "eating"
            row["timestamps"] = [86400 + .25 * index for index in range(16)]
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "overlap in time"):
                prepare_behavior_dataset(library.root, root / "overlap.json")


if __name__ == "__main__":
    unittest.main()
