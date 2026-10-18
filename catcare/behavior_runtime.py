"""Bounded station sequences and an optional asynchronous activity classifier."""
from collections import deque
import hashlib
import threading
import time

from .behavior import ActivityCounter, CLIP_FPS, CLIP_FRAMES, STATION_ACTIVITY, activity_evidence
from .core import box_is_clear_of_zone, pixel_zones
from .diagnostics import redact, gpu_failure


def station_crop(zone, shape):
    height, width = shape[:2]
    left, top, right, bottom = zone
    pad_x, pad_y = (right - left) * .5, (bottom - top) * .5
    return (max(0, int(left - pad_x)), max(0, int(top - pad_y)),
            min(width, int(right + pad_x)), min(height, int(bottom + pad_y)))


class ActivityWorker:
    def __init__(self, manifest, cache, device):
        self.manifest, self.cache, self.device = manifest, cache, device
        self.condition = threading.Condition()
        self.pending, self.results = {}, {}
        self.phase, self.error = "loading", None
        self.stopped = False
        self.thread = threading.Thread(target=self.run, name="cat-activity", daemon=True)
        self.thread.start()

    def submit(self, station, frames, timestamp):
        with self.condition:
            self.pending[station] = (frames, timestamp)
            self.condition.notify()

    def snapshot(self):
        with self.condition:
            return self.phase, self.error, dict(self.results)

    def run(self):
        from .behavior_model import ActivityClassifier
        try:
            try:
                classifier = ActivityClassifier(self.manifest, self.cache, self.device)
            except Exception as error:
                if self.device != "cpu" and gpu_failure(error):
                    self.device = "cpu"
                    classifier = ActivityClassifier(self.manifest, self.cache, "cpu")
                else:
                    raise
            with self.condition:
                self.phase = "ready"
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or self.pending)
                    if self.stopped:
                        return
                    station = next(iter(self.pending))
                    frames, timestamp = self.pending.pop(station)
                if time.time() - timestamp > 3:
                    continue
                try:
                    probabilities = classifier.predict(frames)
                except Exception as error:
                    if self.device != "cpu" and gpu_failure(error):
                        self.device = "cpu"
                        classifier = ActivityClassifier(self.manifest, self.cache, "cpu")
                        probabilities = classifier.predict(frames)
                    else:
                        raise
                with self.condition:
                    self.results[station] = {"timestamp": timestamp, "probabilities": probabilities}
        except Exception as error:
            with self.condition:
                self.phase, self.error = "error", redact(f"{type(error).__name__}: {error}")

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=3)


class BehaviorService:
    def __init__(self, config, library, store, manifest, cache, device):
        self.config, self.library, self.store = config, library, store
        self.model_id = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16] if manifest else "unavailable"
        self.worker = ActivityWorker(manifest, cache, device) if manifest else None
        self.counters = {station: ActivityCounter(resume=store.resume_activity(station, self.model_id))
                         for station in STATION_ACTIVITY if station in config["zones"]}
        self.buffers = {station: deque(maxlen=CLIP_FRAMES) for station in self.counters}
        self.last_frame = self.last_prediction = None
        self.last_sample = {station: 0. for station in self.counters}
        self.last_submit = {station: 0. for station in self.counters}
        self.last_saved = {station: 0. for station in self.counters}
        self.last_result = {station: 0. for station in self.counters}
        self.visible = {station: False for station in self.counters}
        self.latest_prediction_at = 0.
        self.capture_session = None
        self.capture_key = None
        self.capture_stations = ()
        self.captured = 0
        self.note = ""
        self._summary = None
        self._summary_at = 0.
        self.station_states = {station: "waiting" for station in self.counters}

    def record(self, station, events):
        for _, session in events:
            self.store.write_activity(station, session, self.model_id)
            self._summary_at = 0.

    def on_prediction(self, cats, timestamp, shape):
        if self.last_prediction is not None and timestamp <= self.last_prediction:
            return
        self.last_prediction = self.latest_prediction_at = timestamp
        rectangles = pixel_zones(self.config["zones"], shape[1], shape[0])
        for station, counter in self.counters.items():
            self.visible[station] = False
            if len(cats) != 1:
                self.record(station, counter.observe(None, timestamp))
                self.station_states[station] = "uncertain"
            elif box_is_clear_of_zone(cats[0], rectangles[station], margin=.1):
                self.record(station, counter.observe(False, timestamp))
                self.station_states[station] = "away"
            else:
                self.visible[station] = True

    def observe_frame(self, sequence, timestamp, frame, capture_session=None, capture_stations=None, capture_id=None):
        if sequence == self.last_frame:
            return
        self.last_frame = sequence
        selected = tuple(station for station in self.counters
                         if capture_stations is None or station in capture_stations) if capture_session else ()
        capture_key = (capture_session, selected, capture_id)
        if self.capture_key != capture_key:
            for buffer in self.buffers.values():
                buffer.clear()
            for station in selected:
                self.last_saved[station] = 0.
        self.capture_session = capture_session
        self.capture_stations = selected
        self.capture_key = capture_key
        rectangles = pixel_zones(self.config["zones"], frame.shape[1], frame.shape[0])
        for station in self.counters:
            capturing = capture_session is not None and station in selected and self.captured < 120
            recognizing = self.worker is not None and self.visible[station] and timestamp - self.latest_prediction_at <= 2
            if not capturing and not recognizing:
                self.buffers[station].clear()
                continue
            if timestamp - self.last_sample[station] < 1 / CLIP_FPS:
                continue
            buffer = self.buffers[station]
            if buffer and timestamp - buffer[-1][0] > .75:
                buffer.clear()
                self.note = "A sparse camera sequence was discarded; no activity time was inferred."
            self.last_sample[station] = timestamp
            import cv2
            roi = station_crop(rectangles[station], frame.shape)
            left, top, right, bottom = roi
            if right - left < 8 or bottom - top < 8:
                self.note = "Enlarge the station zone so the head and bowl are visible."
                buffer.clear()
                continue
            image = frame[top:bottom, left:right]
            height, width = image.shape[:2]
            if max(height, width) > 640:
                image = cv2.resize(image, (max(1, round(width * 640 / max(height, width))),
                                          max(1, round(height * 640 / max(height, width)))))
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            buffer.append((timestamp, encoded.tobytes()))
            if len(buffer) != CLIP_FRAMES:
                continue
            frames = tuple(row[1] for row in buffer)
            if recognizing and timestamp - self.last_submit[station] >= 1:
                self.worker.submit(station, frames, timestamp)
                self.last_submit[station] = timestamp
            if capturing and timestamp - self.last_saved[station] >= 15:
                try:
                    self.library.add_clip(capture_session, station, frames, [row[0] for row in buffer], roi)
                    self.captured += 1
                    self.note = "Station clips saved privately. Every label requires a human review."
                except ValueError as error:
                    self.note = str(error)
                self.last_saved[station] = timestamp

    def poll(self, now):
        if self.worker:
            phase, error, results = self.worker.snapshot()
            for station, counter in self.counters.items():
                result = results.get(station)
                if not self.visible[station] or now - self.latest_prediction_at > 2:
                    if now - self.latest_prediction_at > 2:
                        self.record(station, counter.observe(None, now))
                        self.station_states[station] = "uncertain"
                    continue
                if result and result["timestamp"] > self.last_result[station] and now - result["timestamp"] <= 3:
                    self.last_result[station] = result["timestamp"]
                    evidence = activity_evidence(station, result["probabilities"])
                    self.record(station, counter.observe(evidence, result["timestamp"]))
                    self.station_states[station] = ("probable_" + STATION_ACTIVITY[station] if evidence is True and counter.session
                                                    else "candidate" if evidence is True else "other" if evidence is False else "uncertain")
                elif result is None or now - result["timestamp"] > 3:
                    self.record(station, counter.observe(None, now))
                    self.station_states[station] = "uncertain"
        else:
            phase, error = "not_available", None
        if self._summary is None or now - self._summary_at >= 1:
            self._summary = self.store.activity_summary(now)
            self._summary_at = now
        return {"phase": phase, "error": error, "stations": dict(self.station_states),
                "summary": self._summary, "captured": self.captured,
                "note": self.note, "collecting": self.capture_session is not None and self.captured < 120,
                "collection_stations": list(self.capture_stations)}

    def interrupt(self):
        for station, counter in self.counters.items():
            session = counter.interrupt()
            if session:
                self.store.write_activity(station, session, self.model_id)

    def stop(self):
        self.interrupt()
        if self.worker:
            self.worker.stop()
