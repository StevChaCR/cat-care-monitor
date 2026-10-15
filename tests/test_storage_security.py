import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from catcare import config
from catcare.core import Visit
from catcare.security import validate_rtsp
from catcare.storage import Store


class SecurityTests(unittest.TestCase):
    def test_authenticated_private_connection(self):
        url = "rtsp://unit:fake@192.168.1.20:554/live0"
        self.assertEqual(validate_rtsp("  " + url + "  "), url)

    def test_public_reserved_and_anonymous_connections_rejected(self):
        for url in ("rtsp://unit:fake@8.8.8.8/live", "rtsp://unit:fake@203.0.113.1/live",
                    "rtsp://unit:fake@0.0.0.0/live", "rtsp://unit:fake@239.1.1.1/live",
                    "rtsp://192.168.1.20/live", "http://unit:fake@192.168.1.20/live"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as error:
                    validate_rtsp(url)
                self.assertNotIn("unit:fake", str(error.exception))

    def test_hostname_cannot_resolve_to_public_address(self):
        resolver = lambda *args: [(2,1,6,"",("192.168.1.20",0)), (2,1,6,"",("8.8.8.8",0))]
        with self.assertRaises(ValueError):
            validate_rtsp("rtsp://unit:fake@camera.local/live", resolver)

    def test_saved_settings_exclude_camera_credentials(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "LOCAL", Path(directory)):
            values = dict(config.DEFAULTS)
            values.update(rtsp="rtsp://unit:fake@192.168.1.20/live", password="fake", username="unit")
            config.save_settings(values)
            saved = json.loads((Path(directory) / "settings.json").read_text())
            self.assertEqual(set(saved), set(config.DEFAULTS))
            self.assertNotIn("rtsp://", json.dumps(saved))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.store = Store(self.path / "visits.sqlite3")

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_rolling_window_clips_duration_and_counts_start_times(self):
        self.store.start_visit("litter_box", "Cat", Visit(99, 120, "completed"))
        self.store.start_visit("litter_box", "Cat", Visit(100, 140, "completed"))
        self.store.start_visit("food", "Cat", Visit(150, 160, "interrupted"))
        summary = self.store.summary(now=200, window_seconds=100)
        self.assertEqual(summary["counts"], {"litter_box": 1, "food": 1})
        self.assertEqual(summary["durations"], {"litter_box": 60, "food": 10})
        self.assertEqual(summary["uncertain"]["food"], 1)

    def test_coverage_merges_overlaps_and_excludes_disconnects(self):
        self.store.mark_coverage(0)
        self.store.mark_coverage(10)
        self.store.break_coverage()
        self.store.mark_coverage(5)
        self.store.mark_coverage(15)
        self.assertEqual(self.store.summary(20, 20)["observed_seconds"], 15)
        self.store.mark_coverage(100)
        self.store.mark_coverage(110)
        self.assertEqual(self.store.summary(110, 20)["observed_seconds"], 10)

    def test_restart_marks_open_visit_interrupted(self):
        self.store.start_visit("water", "Cat", Visit(10, 12))
        self.store.close()
        self.store = Store(self.path / "visits.sqlite3")
        status = self.store.db.execute("SELECT status FROM visits").fetchone()[0]
        self.assertEqual(status, "interrupted")

    def test_csv_preserves_duration_and_escapes_formula_labels(self):
        self.store.start_visit("water", "=test", Visit(10, 12, "completed"))
        self.assertTrue(self.store.export(self.path / "visits.csv"))
        with (self.path / "visits.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        self.assertEqual(rows[1][2], "'=test")
        self.assertEqual(float(rows[1][5]), 2.0)


if __name__ == "__main__":
    unittest.main()
