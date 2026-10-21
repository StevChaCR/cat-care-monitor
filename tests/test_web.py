import copy
import io
import json
import re
import tempfile
import threading
import types
import sys
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from catcare.config import DEFAULTS
from catcare.web_data import ImageLibrary
from catcare.web_runtime import WebRuntime
from catcare.web_training import TrainingJob
from catcare.config import profile_stem
from catcare.detector import Prediction
from catcare.storage import Store
from catcare.tracking import TrackView
from web_app import make_server


class ImageReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.library = ImageLibrary(Path(self.temporary.name))
        self.session = self.library.create_session("Monday afternoon")
        self.image = self.library.add_image(self.session["id"], b"review-fixture", 1920, 1088)

    def test_unreviewed_is_different_from_confirmed_empty(self):
        self.assertFalse(self.library.images()[0]["reviewed"])
        with self.assertRaises(ValueError):
            self.library.save_label(self.image["id"], [])
        self.library.save_label(self.image["id"], [], negative=True)
        row = self.library.images()[0]
        self.assertTrue(row["reviewed"])
        self.assertEqual(row["boxes"], [])

    def test_labels_preserve_full_frame_coordinates(self):
        box = [.4, .6, .8, .9]
        self.library.save_label(self.image["id"], [box])
        for expected, actual in zip(box, self.library.images()[0]["boxes"][0]):
            self.assertAlmostEqual(expected, actual)
        with self.assertRaises(ValueError):
            self.library.save_label(self.image["id"], [[-.1, .2, .5, .6]])

    def test_duplicate_images_cannot_create_independent_sessions(self):
        other = self.library.create_session("Another import of the same frame")
        with self.assertRaises(ValueError):
            self.library.add_image(other["id"], b"review-fixture", 1920, 1088)

    def test_file_references_cannot_escape_the_private_directory(self):
        for key in ("../../settings", "a" * 32 + "_../private", ""):
            with self.assertRaises(ValueError):
                self.library.image_path(key)


class LocalApiTests(unittest.TestCase):
    def test_image_cleanup_is_authenticated_and_preserves_original_labels(self):
        session = self.runtime.library.create_session('Independent occasion')
        row = self.runtime.library.add_image(session['id'], b'private-picture', 640, 480)
        self.runtime.library.save_label(row['id'], [[.1, .2, .5, .8]])
        data = {'ids': [row['id']]}
        self.assertEqual(self.request('POST', '/api/images/trash', data)[0], 403)
        self.assertEqual(self.request('POST', '/api/images/trash', data, authenticated=True)[0], 200)
        images = json.loads(self.request('GET', '/api/library', authenticated=True)[1])
        self.assertEqual(images['images'], [])
        self.assertTrue(images['trash'][0]['reviewed'])
        self.assertEqual(self.request('GET', f"/api/image/{row['id']}?trash=1", authenticated=True), (200, b'private-picture'))
        self.assertEqual(self.request('POST', '/api/images/purge', {'ids': ['../settings']}, authenticated=True)[0], 400)
        self.assertEqual(self.request('POST', '/api/images/restore', data, authenticated=True)[0], 200)
        self.assertTrue(self.runtime.library.images()[0]['reviewed'])
        self.assertEqual(self.request('POST', '/api/images/purge', data, authenticated=True)[0], 400)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.patches = [patch("catcare.config.LOCAL", folder), patch("catcare.web_runtime.LOCAL", folder)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.runtime = WebRuntime()
        self.runtime.config = {**copy.deepcopy(DEFAULTS), "device": "cpu", "crop": False}
        self.server = make_server(self.runtime, 0, training_mode=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        code, body = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.token = re.search(rb'name="catcare-engine" content="([^"]+)"', body).group(1).decode()
        self.assertNotEqual(self.token, "__ENGINE_TOKEN__")

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.runtime.close()

    def request(self, method, path, data=None, headers=None, authenticated=False, raw=False):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        selected = dict(headers or {})
        if authenticated:
            selected["X-CatCare-Token"] = self.token
        body = data if raw else json.dumps(data).encode() if data is not None else None
        if data is not None and not raw:
            selected["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=selected)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    def test_api_requires_session_secret_and_rejects_foreign_origins(self):
        self.assertEqual(self.request("GET", "/api/status")[0], 403)
        self.assertEqual(self.request("GET", "/api/status", authenticated=True)[0], 200)
        self.assertEqual(self.request("GET", "/api/status", authenticated=True,
                                     headers={"Origin": "https://example.org"})[0], 403)
        self.assertEqual(self.request("GET", "/", headers={"Host": "example.org"})[0], 403)

    def test_private_files_are_not_static_assets(self):
        for path in ("/local/settings.json", "/../web_app.py", "/api/image/../../settings.json"):
            self.assertNotEqual(self.request("GET", path, authenticated=True)[0], 200)

    def test_waiting_for_first_video_frame_is_not_an_http_server_error(self):
        code, body = self.request("GET", "/api/frame.jpg", authenticated=True)
        self.assertEqual(code, 204)
        self.assertEqual(body, b"")

    def test_bad_camera_url_is_rejected_without_echoing_its_credentials(self):
        code, body = self.request("POST", "/api/connect", {"url": "rtsp://unit:fake@8.8.8.8/live"}, authenticated=True)
        self.assertEqual(code, 400)
        self.assertNotIn(b"unit:fake", body)
        self.assertFalse(self.runtime.active())

    def test_profile_images_are_separated(self):
        first = self.runtime.library
        first.create_session("First cat")
        code, _ = self.request("POST", "/api/settings", {"cat_label": "Another cat"}, authenticated=True)
        self.assertEqual(code, 200)
        self.assertNotEqual(first.root, self.runtime.library.root)
        self.assertEqual(self.runtime.library.sessions(), [])
        self.assertIs(self.runtime.library, self.runtime.library)

    def test_returning_to_a_profile_restores_its_station_settings(self):
        self.runtime.settings({"cat_label": "Mila", "confidence": .18})
        water = {"water": [.7, .6, .9, .9]}
        self.runtime.zones(water)
        self.runtime.settings({"cat_label": "Another cat", "confidence": .4})
        self.runtime.settings({"cat_label": "Mila"})
        self.assertEqual(self.runtime.config["confidence"], .18)
        self.assertEqual(self.runtime.config["zones"], water)

    def test_incomplete_training_data_does_not_start_a_job(self):
        code, body = self.request("POST", "/api/train", {"epochs": 5}, authenticated=True)
        self.assertEqual(code, 400)
        self.assertIn(b"three distinct recording sessions", body)
        self.assertFalse(self.runtime.training.busy())

    def test_import_review_and_thumbnail_round_trip(self):
        try:
            from PIL import Image
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("Install the application dependencies to test image decoding.")
        code, body = self.request("POST", "/api/sessions", {"name": "Evening camera recording"}, authenticated=True)
        self.assertEqual(code, 200)
        session = json.loads(body)
        photo = io.BytesIO()
        Image.new("RGB", (640, 360), "#99aa88").save(photo, "PNG")
        code, body = self.request("POST", f"/api/upload/{session['id']}", photo.getvalue(), authenticated=True, raw=True)
        self.assertEqual(code, 200)
        row = json.loads(body)
        self.assertEqual((row["width"], row["height"]), (640, 360))
        self.assertEqual(self.request("GET", f"/api/image/{row['id']}")[0], 403)
        code, thumbnail = self.request("GET", f"/api/thumbnail/{row['id']}", authenticated=True)
        self.assertEqual(code, 200)
        with Image.open(io.BytesIO(thumbnail)) as image:
            self.assertEqual(image.width, 360)
        code, _ = self.request("POST", "/api/labels", {"id": row["id"], "boxes": [[.1, .2, .6, .8]]}, authenticated=True)
        self.assertEqual(code, 200)
        self.assertTrue(self.runtime.library.images()[0]["reviewed"])


class TrainingActivationTests(unittest.TestCase):
    def test_only_completed_owned_weights_can_be_activated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned, foreign = root / "job", root / "foreign.pt"
            owned.mkdir()
            foreign.write_bytes(b"not-a-model")
            (owned / "progress.json").write_text(json.dumps({"phase": "completed", "weights": str(foreign)}))
            job = TrainingJob()
            job.folder = owned
            job.process = type("Finished", (), {"poll": lambda self: 0, "returncode": 0})()
            self.assertEqual(job.status()["phase"], "completed")
            self.assertIsNone(job.weights)
            self.assertNotIn("weights", job.status())


class BrowserCameraRuntimeTests(unittest.TestCase):
    def test_headless_runtime_keeps_one_visit_across_loss_and_changed_id(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for simulated video.")
        timeline = [(0, "outside"), (8, "outside"), (10, "water"), (12, "water"),
                    (13, "missing"), (22, "missing"), (26, "water"), (28, "water"),
                    (30, "outside"), (34, "outside")]
        clock = {"index": -1, "time": 100000.0}

        class Steps:
            def is_set(self):
                return clock["index"] >= len(timeline)

            def wait(self, interval):
                clock["index"] += 1
                if clock["index"] >= len(timeline):
                    return True
                clock["time"] = 100000 + timeline[clock["index"]][0]
                return False

        class FakeCamera:
            def __init__(self, *args):
                pass

            def start(self):
                return self

            def latest(self):
                return clock["index"], clock["time"], np.zeros((200, 300, 3), np.uint8)

            def stop(self):
                pass

        class FakeDetector:
            failure = None

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return self

            def latest(self):
                kind = timeline[clock["index"]][1]
                box = (240, 120, 285, 180) if kind == "water" else (20, 20, 50, 50)
                detections = () if kind == "missing" else ((box, .9, 15),)
                tracks = () if kind == "missing" else (
                    TrackView(99 if clock["time"] >= 100026 else 7, box, .9, clock["time"]),)
                return Prediction(clock["index"], clock["time"], (200, 300, 3), detections, 5, tracks)

            def stop(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {**copy.deepcopy(DEFAULTS), "cat_label": "Mila", "crop": False,
                      "zones": {"water": [.8, .5, .95, .9]}}
            model = types.SimpleNamespace(names={15: "cat"})
            with patch("catcare.web_runtime.LOCAL", root), \
                    patch("catcare.camera.Camera", FakeCamera), \
                    patch("catcare.detector.LatestDetector", FakeDetector), \
                    patch("catcare.monitor.initialize_model", return_value=(model, [15], "cpu")), \
                    patch.dict(sys.modules, {"cv2": types.ModuleType("cv2")}), \
                    patch("catcare.web_runtime.time.time", side_effect=lambda: clock["time"]):
                runtime = WebRuntime()
                runtime.stop_event = Steps()
                runtime._run(config, "rtsp://unit:fake@192.168.1.20/live")
            store = Store(root / "data" / f"visits_{profile_stem('Mila')}.sqlite3", recover=False)
            try:
                rows = store.db.execute("SELECT station,entered_at,last_seen_at,status FROM visits").fetchall()
                self.assertEqual(rows, [("water", 100010.0, 100028.0, "uncertain")])
            finally:
                store.close()
