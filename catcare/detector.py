"""A single model worker: camera display never waits for prediction."""
from dataclasses import dataclass
import threading
import time
from .core import pixel_zones
from .display import FrameRate
from .config import LOCAL
from .tracking import CatTrackCache, write_tracker_settings


def processing_crop(rectangles, shape, enabled):
    height, width = shape[:2]
    if not enabled:
        return 0, 0, width, height
    left = min(rect[0] for rect in rectangles.values())
    top = min(rect[1] for rect in rectangles.values())
    right = max(rect[2] for rect in rectangles.values())
    bottom = max(rect[3] for rect in rectangles.values())
    # Include body context instead of tightly cropping bowls or litter-box edges.
    margin_x, margin_y = round((right - left) * 0.75), round((bottom - top) * 0.75)
    return (max(0, left - margin_x), max(0, top - margin_y),
            min(width, right + margin_x), min(height, bottom + margin_y))


@dataclass(frozen=True)
class Prediction:
    sequence: int
    timestamp: float
    image_shape: tuple
    detections: tuple
    fps: float
    tracks: tuple = ()


class LatestDetector:
    """Own one model, process only new frames, and retain only the latest result."""
    def __init__(self, camera, model, config, cat_ids, device, show_all=False):
        self.camera, self.model = camera, model
        self.config, self.cat_ids, self.device = config, cat_ids, device
        self._show_all = show_all
        self._latest = None
        self._failure = None
        self._failure_exception = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cat-detector", daemon=True)
        self._tracker_path = write_tracker_settings(LOCAL / "trackers" / "bytetrack.yaml", config["confidence"])

    def start(self):
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._latest

    @property
    def failure(self):
        with self._lock:
            return self._failure

    @property
    def failure_exception(self):
        """Internal exception for safe diagnostics; never serialize it directly."""
        with self._lock:
            return self._failure_exception

    def set_show_all(self, enabled):
        with self._lock:
            self._show_all = enabled

    def _run(self):
        last_sequence = None
        rate = FrameRate()
        cache = CatTrackCache()
        try:
            while not self._stop.is_set():
                sample = self.camera.latest(after_sequence=last_sequence)
                if sample is None:
                    self._stop.wait(.005)
                    continue
                sequence, timestamp, frame = sample
                last_sequence = sequence
                if time.time() - timestamp > 10:
                    continue
                height, width = frame.shape[:2]
                rectangles = pixel_zones(self.config["zones"], width, height)
                x1, y1, x2, y2 = processing_crop(rectangles, frame.shape, self.config["crop"])
                with self._lock:
                    classes = None if self._show_all else self.cat_ids
                # This is the only thread that calls or mutates this model.
                result = self.model.track(frame[y1:y2, x1:x2], classes=classes,
                    conf=min(.05, self.config["confidence"] / 2), imgsz=self.config["image_size"],
                    device=self.device, verbose=False, persist=True,
                    tracker=str(self._tracker_path))[0]
                detections = []
                observed_tracks = []
                boxes = result.boxes
                ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes.conf.cpu().tolist())
                for box, score, class_id, track_id in zip(boxes.xyxy.cpu().tolist(),
                        boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), ids):
                    class_id = int(class_id)
                    if track_id is None and score < self.config["confidence"]:
                        continue  # Weak unmatched boxes must not create new visits.
                    left, top, right, bottom = box
                    full_box = (left + x1, top + y1, right + x1, bottom + y1)
                    detections.append((full_box, float(score), class_id))
                    if class_id in self.cat_ids and track_id is not None:
                        observed_tracks.append((int(track_id), full_box, float(score)))
                tracks = cache.update(observed_tracks, timestamp)
                if any(label in self.cat_ids for _, _, label in detections) and not observed_tracks:
                    tracks = ()  # Display a strong unassigned box directly instead of a ghost.
                now = time.monotonic()
                rate.record(now)
                prediction = Prediction(sequence, timestamp, tuple(frame.shape),
                                        tuple(detections), rate.value(now), tracks)
                with self._lock:
                    self._latest = prediction
        except Exception as error:
            # Do not print third-party exceptions or potentially sensitive input.
            with self._lock:
                self._failure = "Detection failed. Try a smaller image size or select CPU in setup."
                self._failure_exception = error.with_traceback(None)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
