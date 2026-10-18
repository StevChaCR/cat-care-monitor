"""Thread coordination checks without model downloads or a real camera."""
import importlib.util
import threading
import time
import types
import unittest
from unittest.mock import MagicMock
from catcare.camera import Camera
from catcare.detector import LatestDetector
from catcare.display import FrameRate


class Tensor:
    def __init__(self, values):
        self.values = values
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
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
        input_sizes = []
        tensors = []
        model = MagicMock()

        def predict(image, **kwargs):
            calls.append(image.shape)
            input_sizes.append(kwargs['imgsz'])
            if len(calls) == 1:
                started.set()
                if not release.wait(2):
                    raise RuntimeError('Test did not release blocked inference')
            data = Tensor([[15, 20, 100, 140, 7, .9, 15]])
            tensors.append(data)
            boxes = types.SimpleNamespace(data=data, is_track=True)
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
            detector.set_image_size(640)
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
            self.assertEqual(input_sizes, [960, 640])
            self.assertTrue(all(tensor.cpu_calls == 1 for tensor in tensors))
            self.assertGreater(latest.processing_ms, 0)
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

    def test_strong_unmatched_box_retains_original_coordinates_and_weak_box_is_ignored(self):
        import numpy as np
        camera = Camera('rtsp://unit:fake@192.168.1.20/live', None)
        camera._latest = (1, time.time(), np.zeros((200, 300, 3), dtype=np.uint8))
        model = MagicMock()
        model.track.return_value = [types.SimpleNamespace(boxes=types.SimpleNamespace(
            data=Tensor([[15, 20, 100, 140, .9, 15], [30, 40, 120, 150, .1, 15]]), is_track=False))]
        config = {'zones': {}, 'crop': False, 'confidence': .25, 'image_size': 640}
        detector = LatestDetector(camera, model, config, [15], 'cpu').start()
        try:
            deadline = time.monotonic() + 2
            while detector.latest() is None and detector.failure is None and time.monotonic() < deadline:
                time.sleep(.005)
            result = detector.latest()
            self.assertIsNotNone(result)
            self.assertEqual(result.detections, (((15, 20, 100, 140), .9, 15),))
            self.assertEqual(result.tracks, ())
        finally:
            detector.stop()


class LowFrameRateTests(unittest.TestCase):
    def test_fractional_frame_rates_remain_measurable_and_stop_when_stale(self):
        rate = FrameRate()
        rate.record(0)
        rate.record(4)
        self.assertAlmostEqual(rate.value(4), .25)
        rate.record(8)
        self.assertAlmostEqual(rate.value(8), .25)
        self.assertEqual(rate.value(11), 0)
