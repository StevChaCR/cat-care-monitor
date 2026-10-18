import json
import re
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

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
        for path in ("/api/behavior/train", "/api/behavior/collect", "/api/train", "/api/sessions"):
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

    def test_no_behavior_model_does_not_generate_eating_or_drinking_counts(self):
        self.setup_server(False)
        code, body = self.request("GET", "/api/status")
        row = json.loads(body)
        self.assertEqual(code, 200)
        self.assertFalse(row["activity_model_available"])
        self.assertIsNone(row["behavior"]["summary"])


if __name__ == "__main__":
    unittest.main()
