import json
import re
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch, MagicMock

from catcare.web_runtime import WebRuntime
from web_app import make_server


class BehaviorApiTests(unittest.TestCase):
    def setup_server(self, training):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        for target in ("catcare.config.LOCAL", "catcare.web_runtime.LOCAL"):
            item = patch(target, folder); item.start(); self.addCleanup(item.stop)
        self.runtime = WebRuntime()
        self.server = make_server(self.runtime, 0, training_mode=training)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), thread.join(3), self.runtime.close()))
        code, page = self.request("GET", "/", authenticated=False)
        self.assertEqual(code, 200)
        self.token = re.search(rb'name="catcare-engine" content="([^\"]+)"', page).group(1).decode()
        return page

    def request(self, method, path, data=None, authenticated=True):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"X-CatCare-Token": self.token} if authenticated else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, json.dumps(data) if data is not None else None, headers)
        response = connection.getresponse(); code, body = response.status, response.read(); connection.close()
        return code, body

    def test_normal_mode_disables_training_endpoints_even_with_valid_token(self):
        page = self.setup_server(False)
        self.assertIn(b'name="catcare-training" content="false"', page)
        for path in ("/api/library", "/api/behavior/library", "/api/behavior/frame/anything/0"):
            self.assertEqual(self.request("GET", path)[0], 403)
        for path in ("/api/behavior/train", "/api/behavior/collect", "/api/behavior/clips/trash", "/api/behavior/clips/restore",
                     "/api/behavior/clips/purge", "/api/train", "/api/train/bundle", "/api/sessions", "/api/performance"):
            self.assertEqual(self.request("POST", path, {})[0], 403)
        self.assertEqual(self.request("GET", "/api/status")[0], 200)

    def test_lab_clip_labels_are_private_and_profile_scoped(self):
        page = self.setup_server(True)
        self.assertIn(b'name="catcare-training" content="true"', page)
        self.assertEqual(self.request("GET", "/api/behavior/library", authenticated=False)[0], 403)
        code, body = self.request("POST", "/api/behavior/sessions", {"name": "Saturday evening"})
        self.assertEqual(code, 200)
        session = json.loads(body)
        clip = self.runtime.behavior_library.add_clip(session["id"], "food", tuple(f"jpeg-{i}".encode() for i in range(16)),
            [10 + .25 * i for i in range(16)], [0, 0, 640, 480])
        self.assertEqual(self.request("POST", "/api/behavior/labels", {"id": clip["id"], "label": "other"})[0], 200)
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/0")[0], 200)
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/99")[0], 400)
        self.runtime.settings({"cat_label": "Another profile"})
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/0")[0], 400)

    def test_lab_cleanup_requires_authentication_and_restores_original_frames(self):
        self.setup_server(True)
        session = self.runtime.behavior_library.create_session("Evening")
        clip = self.runtime.behavior_library.add_clip(session["id"], "water", tuple(f"jpeg-{i}".encode() for i in range(16)),
            [10 + .25 * i for i in range(16)], [0, 0, 640, 480])
        payload = {"ids": [clip["id"]]}
        self.assertEqual(self.request("POST", "/api/behavior/clips/trash", payload, authenticated=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/behavior/clips/trash", payload)[0], 200)
        data = json.loads(self.request("GET", "/api/behavior/library")[1])
        self.assertEqual(data["clips"], [])
        self.assertEqual(len(data["trash"]), 1)
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/0")[0], 400)
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/0?trash=1"), (200, b"jpeg-0"))
        self.assertEqual(self.request("POST", "/api/behavior/clips/restore", payload)[0], 200)
        self.assertEqual(self.request("GET", f"/api/behavior/frame/{clip['id']}/0"), (200, b"jpeg-0"))
        self.assertEqual(self.request("POST", "/api/behavior/clips/purge", payload)[0], 400)
        self.assertEqual(self.request("POST", "/api/behavior/clips/trash", payload)[0], 200)
        self.runtime.settings({"cat_label": "Another profile"})
        self.assertEqual(self.request("POST", "/api/behavior/clips/purge", payload)[0], 400)

    def test_collection_endpoint_selects_only_the_requested_configured_station(self):
        self.setup_server(True)
        self.runtime.config["zones"] = {"food": [.05, .05, .4, .4], "water": [.6, .6, .95, .95]}
        session = self.runtime.behavior_library.create_session("Same real period")
        with patch.object(self.runtime, "active", return_value=True):
            self.assertEqual(self.request("POST", "/api/behavior/collect", {"enabled": True, "session": session["id"], "station": "water"})[0], 200)
            self.assertEqual(self.runtime.behavior_capture_stations, ("water",))
            data = json.loads(self.request("GET", "/api/status")[1])
            self.assertEqual(data["behavior"]["collection_stations"], ["water"])
            self.assertEqual(self.request("POST", "/api/behavior/collect", {"enabled": True, "session": session["id"], "station": "litter_box"})[0], 400)
            self.assertEqual(self.request("POST", "/api/behavior/collect", {"enabled": False})[0], 200)
            self.assertIsNone(self.runtime.behavior_capture_session)

    def test_no_behavior_model_does_not_generate_eating_or_drinking_counts(self):
        self.setup_server(False)
        code, body = self.request("GET", "/api/status")
        row = json.loads(body)
        self.assertEqual(code, 200)
        self.assertFalse(row["activity_model_available"])
        self.assertIsNone(row["behavior"]["summary"])

    def test_fast_inference_changes_size_live_without_changing_the_trained_model(self):
        self.setup_server(True)
        model = self.runtime.config["model"]
        detector = MagicMock()
        self.runtime.detector = detector
        with patch.object(self.runtime, "active", return_value=True):
            self.assertEqual(self.request("POST", "/api/performance", {"image_size": 640})[0], 200)
        detector.set_image_size.assert_called_once_with(640)
        self.assertEqual(self.runtime.config["image_size"], 640)
        self.assertEqual(self.runtime.config["model"], model)
        for size in (True, "640", 640.5, 0, 999999):
            self.assertEqual(self.request("POST", "/api/performance", {"image_size": size})[0], 400)


if __name__ == "__main__":
    unittest.main()
