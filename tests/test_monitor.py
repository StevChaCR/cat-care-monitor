"""Simulated camera/model integration; no actual camera or weights are used."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from catcare import monitor
from catcare.detector import Prediction, processing_crop
from catcare.tracking import CatTrackCache
from catcare.config import profile_stem
from catcare.storage import Store

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


@unittest.skipUnless(HAS_NUMPY, "NumPy is needed for simulated frame integration")
class MonitorTests(unittest.TestCase):
    def run_simulation(self, root, preview, hold_prediction=False, timeline=None):
        import numpy as np
        timeline = timeline or [(0, 'outside'), (0, 'outside'), (8, 'outside'), (10, 'litter'),
                                (11, 'litter'), (12, 'litter'), (13, 'outside'), (21, 'outside')]

        class FakeCamera:
            current_time = 100000.0
            current_state = 'outside'

            def __init__(self, *args):
                self.index = -1

            def start(self):
                return self

            def latest(self):
                self.index += 1
                offset, state = timeline[min(self.index, len(timeline) - 1)]
                FakeCamera.current_time = 100000 + offset
                FakeCamera.current_state = state
                return self.index, FakeCamera.current_time, np.zeros((200, 300, 3), dtype=np.uint8)

            def stop(self):
                pass

        model = MagicMock()
        model.names = {15: "cat"}

        def predict(frame, **kwargs):
            location = {'litter': [15,20,100,140], 'water': [240,60,285,135],
                        'outside': [240,5,285,35]}.get(FakeCamera.current_state)
            boxes = types.SimpleNamespace(xyxy=Tensor([location] if location else []),
                                          conf=Tensor([.9] if location else []), cls=Tensor([15] if location else []))
            return [types.SimpleNamespace(boxes=boxes)]

        model.predict.side_effect = predict

        class FakeDetector:
            failure = None

            def __init__(self, camera, model, config, cat_ids, device, show_all=False):
                self.camera, self.model, self.config = camera, model, config
                self.cat_ids, self.device, self.show_all = cat_ids, device, show_all
                self.previous = None
                self.cache = CatTrackCache()

            def start(self):
                return self

            def latest(self):
                if hold_prediction and self.previous and self.previous.timestamp >= 100012:
                    return self.previous
                frame = np.zeros((200, 300, 3), dtype=np.uint8)
                rectangles = monitor.pixel_zones(self.config['zones'], 300, 200)
                x1, y1, x2, y2 = processing_crop(rectangles, frame.shape, self.config['crop'])
                result = self.model.predict(frame[y1:y2, x1:x2], classes=None if self.show_all else self.cat_ids,
                    conf=self.config['confidence'], imgsz=self.config['image_size'], device=self.device, verbose=False)[0]
                detections = tuple(((a + x1, b + y1, c + x1, d + y1), score, int(label))
                    for (a, b, c, d), score, label in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()))
                # ID changes do not represent a physical exit or a new visit.
                track_id = 99 if FakeCamera.current_time >= 100026 else 7
                tracks = self.cache.update([(track_id, box, score) for box, score, _ in detections], FakeCamera.current_time)
                self.previous = Prediction(self.camera.index, FakeCamera.current_time, frame.shape, detections, 5, tracks)
                return self.previous

            def set_show_all(self, enabled):
                self.show_all = enabled

            def stop(self):
                pass

        cv = types.ModuleType("cv2")
        cv.WINDOW_NORMAL = 0
        cv.WINDOW_AUTOSIZE = 1
        cv.WND_PROP_VISIBLE = 4
        cv.FONT_HERSHEY_SIMPLEX = 0
        cv.LINE_AA = 16
        cv.INTER_AREA = 3
        cv.INTER_CUBIC = 2
        cv.error = RuntimeError
        cv.getWindowProperty = MagicMock(return_value=1)
        for name in ("namedWindow", "moveWindow", "rectangle", "putText", "imshow", "destroyAllWindows"):
            setattr(cv, name, MagicMock())
        # Mutate display images so a saved frame would reveal overlay leakage.
        cv.rectangle.side_effect = lambda image, *args: image.__setitem__((0, 0), (0, 255, 0))
        cv.putText.side_effect = lambda image, *args: image.__setitem__((0, 0), (255, 255, 255))
        cv.resize = MagicMock(side_effect=lambda image, size, **kwargs: np.zeros((size[1], size[0], 3), dtype=np.uint8))
        cv.getTextSize = lambda text, font, scale, thickness: ((round(len(text) * 11 * scale), 18), 6)
        cv.waitKey = MagicMock(side_effect=[ord('s') if preview else 0] + [0] * (len(timeline) - 3) + [ord('q')])
        cv.imwrite = MagicMock(return_value=True)
        config = {"cat_label": "Mila", "model": "yolo26n.pt", "device": "cpu",
                  "confidence": .25, "image_size": 960, "crop": True,
                  "zones": {"litter_box": [.05,.5,.35,.9], "food": [.5,.5,.7,.9], "water": [.8,.5,.95,.9]}}
        output = io.StringIO()
        with patch.object(monitor, "LOCAL", root), patch.object(monitor, "Camera", FakeCamera), \
                patch.object(monitor, "LatestDetector", FakeDetector), \
                patch.object(monitor, "initialize_model", return_value=(model, [15], "cpu")), \
                patch.object(monitor, "screen_bounds", return_value=(960, 700)), \
                patch.dict(sys.modules, {"cv2": cv}), \
                patch.object(monitor.time, "time", side_effect=lambda: FakeCamera.current_time), \
                patch.object(monitor.time, "sleep"), contextlib.redirect_stdout(output):
            monitor.run_monitor(config, "rtsp://unit:fake@192.168.1.20/live", preview=preview)
        self.assertNotIn("rtsp://", output.getvalue())
        self.assertEqual(model.predict.call_args.args[0].shape, (160, 300, 3))
        self.assertEqual(cv.imshow.call_args.args[1].shape, (700, 960, 3))
        return cv

    def test_real_visit_store_with_simulated_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_simulation(root, preview=False)
            db_path = root / "data" / f"visits_{profile_stem('Mila')}.sqlite3"
            store = Store(db_path, recover=False)
            try:
                rows = store.db.execute("SELECT station,entered_at,last_seen_at,status FROM visits").fetchall()
                self.assertEqual(rows, [("litter_box", 100010.0, 100012.0, "completed")])
                self.assertEqual(store.summary(100021)["counts"], {"litter_box": 1})
                self.assertEqual(store.summary(100021)["durations"], {"litter_box": 2.0})
            finally:
                store.close()
            self.assertTrue(list((root / "reports").glob("*.csv")))

    def test_preview_does_not_record_visits_and_saves_full_raw_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(monitor, "Store", side_effect=AssertionError("Preview opened a visit database")):
                cv = self.run_simulation(root, preview=True)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "reports").exists())
            self.assertEqual(cv.imwrite.call_args.args[1].shape, (200, 300, 3))
            self.assertFalse(cv.imwrite.call_args.args[1].any())
            self.assertTrue(str(cv.imwrite.call_args.args[0]).startswith(str(root / "captures")))

    def test_repainting_old_prediction_does_not_extend_visit_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_simulation(root, preview=False, hold_prediction=True)
            store = Store(root / "data" / f"visits_{profile_stem('Mila')}.sqlite3", recover=False)
            try:
                rows = store.db.execute("SELECT entered_at,last_seen_at,status FROM visits").fetchall()
                self.assertEqual(rows, [(100010.0, 100012.0, "interrupted")])
            finally:
                store.close()

    def test_water_visit_is_not_duplicated_by_dropout_or_changed_tracking_id(self):
        timeline = [(0, 'outside'), (0, 'outside'), (8, 'outside'), (10, 'water'),
                    (11, 'water'), (12, 'water'), (13, 'missing'), (22, 'missing'),
                    (26, 'water'), (27, 'water'), (28, 'water'), (30, 'outside'), (34, 'outside')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_simulation(root, preview=False, timeline=timeline)
            store = Store(root / 'data' / f"visits_{profile_stem('Mila')}.sqlite3", recover=False)
            try:
                rows = store.db.execute('SELECT station,entered_at,last_seen_at,status FROM visits').fetchall()
                self.assertEqual(rows, [('water', 100010.0, 100028.0, 'uncertain')])
                self.assertEqual(store.summary(100034)['counts'], {'water': 1})
            finally:
                store.close()

    def test_observed_departure_and_return_create_two_water_visits(self):
        timeline = [(0, 'outside'), (0, 'outside'), (8, 'outside'), (10, 'water'),
                    (12, 'water'), (13, 'outside'), (17, 'outside'), (20, 'water'),
                    (22, 'water'), (23, 'outside'), (27, 'outside')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_simulation(root, preview=False, timeline=timeline)
            store = Store(root / 'data' / f"visits_{profile_stem('Mila')}.sqlite3", recover=False)
            try:
                rows = store.db.execute('SELECT station,entered_at,last_seen_at,status FROM visits ORDER BY id').fetchall()
                self.assertEqual(rows, [('water', 100010.0, 100012.0, 'completed'),
                                        ('water', 100020.0, 100022.0, 'completed')])
                self.assertEqual(store.summary(100027)['counts'], {'water': 2})
            finally:
                store.close()


class WindowTests(unittest.TestCase):
    def test_closing_window_requests_exit(self):
        cv = types.SimpleNamespace(waitKey=lambda _: 0, getWindowProperty=lambda *args: 0,
                                   WND_PROP_VISIBLE=4, error=RuntimeError)
        self.assertEqual(monitor.read_key(cv), ord("q"))
