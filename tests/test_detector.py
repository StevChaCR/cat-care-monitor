"""Thread coordination checks without model downloads or a real camera."""
import importlib.util
import threading
import time
import types
import unittest
from unittest.mock import MagicMock
from catcare.camera import Camera
from catcare.detector import LatestDetector


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy is needed for worker checks')
class DetectorTests(unittest.TestCase):
    def test_slow_prediction_does_not_block_reads_and_skips_old_camera_frames(self):
        import numpy as np
        camera = Camera('rtsp://unit:fake@192.168.1.20/live', None)
        captured_at = time.time()
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        camera._latest = (1, captured_at, frame)
        started, release = threading.Event(), threading.Event()
        calls = []
        model = MagicMock()

        def predict(image, **kwargs):
            calls.append(image.shape)
            if len(calls) == 1:
                started.set()
                if not release.wait(2):
                    raise RuntimeError('Test did not release blocked inference')
            boxes = types.SimpleNamespace(xyxy=Tensor([[15, 20, 100, 140]]), conf=Tensor([.9]), cls=Tensor([15]), id=Tensor([7]))
            return [types.SimpleNamespace(boxes=boxes)]

        model.track.side_effect = predict
        config = {'zones': {'litter_box': [.05, .5, .35, .9]}, 'crop': False,
                  'confidence': .25, 'image_size': 960}
        detector = LatestDetector(camera, model, config, [15], 'cpu').start()
        try:
            self.assertTrue(started.wait(1))
            self.assertIsNone(detector.latest())  # Returns while predict is blocked.
            with camera._lock:
                camera._latest = (2, captured_at + .01, frame)
                camera._latest = (3, captured_at + .02, frame)
            self.assertEqual(camera.latest()[0], 3)
            release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                latest = detector.latest()
                if latest and latest.sequence == 3:
                    break
                time.sleep(.005)
            self.assertEqual(latest.sequence, 3)
            self.assertEqual(latest.timestamp, captured_at + .02)
            self.assertEqual(latest.detections[0][0], (15, 20, 100, 140))
            self.assertEqual(latest.tracks[0].track_id, 7)
            self.assertTrue(model.track.call_args.kwargs['persist'])
            self.assertLess(model.track.call_args.kwargs['conf'], config['confidence'])
            self.assertEqual(len(calls), 2)
            self.assertIsNone(camera.latest(after_sequence=3))
        finally:
            release.set()
            detector.stop()

    def test_prediction_failure_is_reported_without_third_party_message(self):
        import numpy as np
        camera = Camera('rtsp://unit:fake@192.168.1.20/live', None)
        camera._latest = (1, time.time(), np.zeros((100, 100, 3), dtype=np.uint8))
        model = MagicMock()
        model.track.side_effect = RuntimeError('rtsp://unit:fake@192.168.1.20/live')
        config = {'zones': {'food': [.1, .1, .5, .5]}, 'crop': False,
                  'confidence': .25, 'image_size': 640}
        detector = LatestDetector(camera, model, config, [15], 'cpu').start()
        try:
            deadline = time.monotonic() + 2
            while detector.failure is None and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertIn('Detection failed', detector.failure)
            self.assertNotIn('rtsp://', detector.failure)
            self.assertNotIn('unit:fake', detector.failure)
        finally:
            detector.stop()
