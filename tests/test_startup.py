"""Camera-first startup, preview survival, GPU recovery, and safe diagnostics."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare.config import DEFAULTS
from catcare.detector import Prediction
from catcare.diagnostics import diagnostic, gpu_failure, redact
from catcare.web_runtime import WebRuntime


class DiagnosticTests(unittest.TestCase):
    def test_full_urls_and_separately_echoed_credentials_are_redacted(self):
        source = "rtsp://unit:fake@192.168.1.20/live"
        row = diagnostic("model", RuntimeError(f"Failed with {source}; user=unit password=fake"), source)
        for private in (source, "rtsp://", "unit", "fake", "192.168.1.20"):
            self.assertNotIn(private, row["detail"])

    def test_percent_encoded_credentials_and_exception_causes_are_redacted(self):
        from urllib.parse import quote
        username, password = "unit user", "fake@pass"
        source = "rtsp://" + quote(username) + ":" + quote(password, safe="") + "@192.168.1.20/live"
        original = ValueError(f"{username} {password}")
        outer = RuntimeError("Model startup failed")
        outer.__cause__ = original
        row = diagnostic("model", outer, source)
        self.assertNotIn(username, row["detail"])
        self.assertNotIn(password, row["detail"])
        self.assertNotIn("https://", redact("Download failed: https://example.org/file?token=private"))

    def test_model_dependency_and_gpu_errors_are_distinguished(self):
        missing = diagnostic("model", ModuleNotFoundError("No module named 'lap'"))
        self.assertEqual(missing["code"], "dependency_unavailable")
        self.assertIn("lap", missing["detail"])
        self.assertTrue(gpu_failure(RuntimeError("CUDA error: no kernel image is available")))
        self.assertFalse(gpu_failure(RuntimeError("No camera frames received")))


class StartupRecoveryTests(unittest.TestCase):
    def simulate(self, root, initial_error=None, inference_error=None, no_camera=False):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for startup frame simulation.")
        runtime = WebRuntime()
        config = {**copy.deepcopy(DEFAULTS), "device": "auto", "crop": False, "zones": {}}
        phases, preview_bytes, loaded_devices, detector_devices = [], [], [], []
        camera_events = []
        frame = np.zeros((100, 160, 3), np.uint8)

        class Camera:
            status = "Unavailable" if no_camera else "Live"
            failure_exception = RuntimeError("OpenCV FFmpeg video backend is unavailable.") if no_camera else None

            def __init__(self, *args):
                camera_events.append("created")

            def start(self):
                camera_events.append("started")
                return self

            def latest(self):
                return None if no_camera else (1, time.time(), frame.copy())

            def stop(self):
                camera_events.append("stopped")

        class Steps:
            index = 0

            def is_set(self):
                return self.index >= 4

            def wait(self, interval):
                self.index += 1
                phases.append(runtime.state["phase"])
                if runtime.camera is not None:
                    preview_bytes.append(runtime.jpeg())
                return self.is_set()

        class Detector:
            def __init__(self, camera, model, config, cat_ids, device, **kwargs):
                self.device = device
                detector_devices.append(device)
                self.failure_exception = inference_error if device == "0" else None
                self.failure = "Detection failed" if self.failure_exception else None

            def start(self):
                return self

            def latest(self):
                return Prediction(1, time.time(), (100, 160, 3), (), 5)

            def stop(self):
                pass

        def load(config):
            self.assertIn("started", camera_events, "Camera preview must start before model loading")
            loaded_devices.append(config["device"])
            if initial_error and config["device"] != "cpu":
                raise initial_error
            return types.SimpleNamespace(names={15: "cat"}), [15], "cpu" if config["device"] == "cpu" else "0"

        cv = types.ModuleType("cv2")
        cv.IMWRITE_JPEG_QUALITY = 1
        cv.imencode = lambda *args: (True, types.SimpleNamespace(tobytes=lambda: b"preview-frame"))
        output = io.StringIO()
        runtime.stop_event = Steps()
        with patch("catcare.web_runtime.LOCAL", root), \
                patch("catcare.camera.Camera", Camera), \
                patch("catcare.detector.LatestDetector", Detector), \
                patch("catcare.monitor.initialize_model", side_effect=load), \
                patch.dict(sys.modules, {"cv2": cv}), contextlib.redirect_stdout(output):
            runtime._run(config, "rtsp://unit:fake@192.168.1.20/live")
        self.assertNotIn("rtsp://", output.getvalue())
        return runtime, phases, preview_bytes, loaded_devices, detector_devices

    def test_dependency_failure_keeps_camera_preview_and_records_specific_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, phases, frames, devices, _ = self.simulate(root, initial_error=ModuleNotFoundError("No module named 'lap'"))
            self.assertIn("preview", phases)
            self.assertIn(b"preview-frame", frames)
            self.assertEqual(devices, ["auto"])
            self.assertEqual(runtime.state["error"]["code"], "dependency_unavailable")
            saved = json.loads((root / "diagnostics" / "latest.json").read_text())
            self.assertIn("lap", saved["detail"])

    def test_gpu_inference_failure_retries_cpu_without_closing_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, phases, frames, devices, detected = self.simulate(Path(folder), inference_error=RuntimeError("CUDA error: no kernel image is available"))
            self.assertEqual(devices, ["auto", "cpu"])
            self.assertEqual(detected, ["0", "cpu"])
            self.assertIn(b"preview-frame", frames)
            self.assertIn("live", phases)
            self.assertIn("CPU", runtime.state["notice"])

    def test_gpu_model_loading_failure_also_retries_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, _, frames, devices, detected = self.simulate(Path(folder), initial_error=RuntimeError("CUDA out of memory"))
            self.assertEqual(devices, ["auto", "cpu"])
            self.assertEqual(detected, ["cpu"])
            self.assertIn(b"preview-frame", frames)

    def test_unavailable_camera_is_reported_without_loading_the_model(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime, _, _, devices, _ = self.simulate(Path(folder), no_camera=True)
            self.assertEqual(devices, [])
            self.assertEqual(runtime.state["error"]["code"], "opencv_ffmpeg_unavailable")
            self.assertEqual(runtime.state["phase"], "error")
