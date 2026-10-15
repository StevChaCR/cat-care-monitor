"""Headless local camera runtime for the browser interface."""
import copy
import json
import threading
import time
from datetime import datetime, timezone

from .config import DEFAULTS, LOCAL, load_settings, profile_stem, save_settings, validate_settings
from .core import VisitCounter, assign_station, pixel_zones, station_presence, validate_zones
from .display import FrameRate
from .diagnostics import diagnostic, gpu_failure, save_diagnostic
from .security import validate_rtsp
from .storage import Store
from .web_data import ImageLibrary
from .web_training import TrainingJob


class WebRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self._libraries = {}
        self.config = load_settings()
        self.config["crop"] = False
        self.camera = None
        self.worker = None
        self.stop_event = threading.Event()
        self.training = TrainingJob()
        self.collecting = False
        self.collection_session = None
        self.collection_count = 0
        self.encoded_frame = None
        self.encoded_sequence = None
        self.frame_lock = threading.Lock()
        self.state = {"phase": "offline", "message": "Connect your camera to begin.",
                      "detections": [], "tracks": [], "ai_fps": 0, "video_fps": 0,
                      "summary": None, "processor": None, "frame_shape": None,
                      "error": None, "notice": None}

    @property
    def profile(self):
        return LOCAL / "web" / profile_stem(self.config["cat_label"])

    @property
    def library(self):
        with self.lock:
            path = self.profile / "sessions"
            if path not in self._libraries:
                self._libraries[path] = ImageLibrary(path)
            return self._libraries[path]

    def active(self):
        return self.worker is not None and self.worker.is_alive()

    def status(self):
        with self.lock:
            return {**copy.deepcopy(self.state), "config": copy.deepcopy(self.config),
                    "running": self.active(),
                    "collecting": self.collecting, "collected": self.collection_count,
                    "training": self.training.status()}

    def settings(self, values):
        with self.lock:
            if self.active() or self.training.busy():
                raise ValueError("Stop the camera and training before changing the profile.")
            allowed = {key: values[key] for key in ("cat_label", "device", "confidence", "image_size") if key in values}
            changed_profile = allowed.get("cat_label", self.config["cat_label"]) != self.config["cat_label"]
            base = self.config
            if changed_profile:
                self._save_config()
                label = str(allowed["cat_label"]).strip()
                saved = LOCAL / "web" / profile_stem(label) / "settings.json"
                base = {**DEFAULTS, "cat_label": label, "zones": {}}
                if saved.is_file():
                    previous = json.loads(saved.read_text(encoding="utf-8"))
                    base.update({key: previous[key] for key in DEFAULTS if key in previous})
                self.training = TrainingJob()
                self.collection_session = None
                self.state.update(summary=None, detections=[], tracks=[])
            candidate = {**base, **allowed, "crop": False}
            if "model" in values:
                if values["model"] not in ("yolo26n.pt", "yolo26s.pt"):
                    raise ValueError("Choose a built-in detector or activate your completed training run.")
                candidate["model"] = values["model"]
            candidate = validate_settings(candidate)
            if candidate["zones"]:
                validate_zones(candidate["zones"])
            self.config = candidate
            self._save_config()

    def _save_config(self):
        save_settings(self.config)
        self.profile.mkdir(parents=True, exist_ok=True)
        path = self.profile / "settings.json"
        temporary = path.with_suffix(".tmp")
        public = {key: self.config[key] for key in DEFAULTS}
        temporary.write_text(json.dumps(public, indent=2), encoding="utf-8")
        temporary.replace(path)

    def zones(self, zones):
        if self.active():
            raise ValueError("Disconnect the camera before saving station zones.")
        validate_zones(zones)
        with self.lock:
            self.config["zones"] = zones
            self._save_config()

    def connect(self, source):
        with self.lock:
            if self.active() or self.training.busy():
                raise ValueError("Stop the current camera or training job first.")
            source = validate_rtsp(source)
            self.stop_event = threading.Event()
            self.state.update(phase="connecting", message="Connecting to the camera…",
                              detections=[], tracks=[], summary=None, error=None, notice=None,
                              ai_fps=0, video_fps=0, processor=None)
            self.encoded_frame = self.encoded_sequence = None
            config = copy.deepcopy(self.config)
            self.worker = threading.Thread(target=self._run, args=(config, source), daemon=True)
            self.worker.start()

    def disconnect(self):
        with self.lock:
            self.stop_event.set()
            self.collecting = False
            worker = self.worker
        if worker is not None:
            worker.join(timeout=15)
            if worker.is_alive():
                raise ValueError("The camera is still stopping. Please wait a moment.")

    def collect(self, enabled):
        with self.lock:
            if enabled and not self.active():
                raise ValueError("Connect the camera before collecting learning images.")
            if enabled and self.collection_session is None:
                name = datetime.now(timezone.utc).strftime("Camera · %Y-%m-%d %H:%M UTC")
                self.collection_session = self.library.create_session(name)["id"]
                self.collection_count = 0
            self.collecting = bool(enabled)

    def jpeg(self):
        with self.frame_lock:
            camera = self.camera
            sample = camera.latest() if camera else None
            if sample is None or time.time() - sample[1] > 10:
                return None
            import cv2
            sequence, _, frame = sample
            if sequence != self.encoded_sequence:
                height, width = frame.shape[:2]
                if width > 1280:
                    frame = cv2.resize(frame, (1280, round(height * 1280 / width)), interpolation=cv2.INTER_AREA)
                ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    return None
                self.encoded_frame = image.tobytes()
                self.encoded_sequence = sequence
            return self.encoded_frame

    def start_training(self, epochs):
        if self.active():
            raise ValueError("Disconnect the camera before training so it can use the available processor.")
        device = self.config["device"]
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self.training.start(self.profile, self.library.root, self.config["model"], epochs,
                            min(self.config["image_size"], 960), device)

    def activate_model(self):
        with self.lock:
            if self.active() or self.training.busy():
                raise ValueError("Stop the camera before changing models.")
            status = self.training.status()
            if status.get("phase") != "completed" or self.training.weights is None:
                raise ValueError("Complete and review a training run first.")
            self.config["model"] = str(self.training.weights.resolve())
            self._save_config()

    def _run(self, config, source):
        from .camera import Camera
        from .detector import LatestDetector
        from .monitor import initialize_model
        detector = store = camera = None
        model, cat_ids, device = None, [], None
        private_source = source
        stage = "camera"
        cpu_attempted = config["device"] == "cpu"
        counters = {name: VisitCounter() for name in config["zones"]}
        active_ids = {}
        last_prediction = None
        last_prediction_time = None
        last_sequence = None
        last_summary = last_capture = last_export = 0
        video_rate = FrameRate()
        collection_previous = None
        disconnected = False

        def report(stage, error):
            row = diagnostic(stage, error, private_source)
            save_diagnostic(LOCAL / "diagnostics", row)
            with self.lock:
                self.state["error"] = row
            return row

        def start_detector(requested):
            nonlocal model, cat_ids, device, detector
            model, cat_ids, device = initialize_model(requested)
            detector = LatestDetector(camera, model, requested, cat_ids, device, show_all=False).start()

        def detector_failed(stage, error):
            nonlocal detector, model, cpu_attempted, last_prediction
            failed_device = device or config["device"]
            report(stage, error)
            if detector:
                detector.stop()
            detector, model, last_prediction = None, None, None
            with self.lock:
                self.state.update(detections=[], tracks=[], ai_fps=0, notice=None)
            if not cpu_attempted and failed_device != "cpu" and gpu_failure(error):
                cpu_attempted = True
                with self.lock:
                    self.state.update(phase="loading_model", message="GPU processing failed. Retrying detection on CPU…")
                try:
                    start_detector({**config, "device": "cpu"})
                    with self.lock:
                        self.state["notice"] = "GPU processing failed; detection is running on CPU."
                    return
                except Exception as cpu_error:
                    report("model", cpu_error)
                    detector = model = None
            with self.lock:
                self.state.update(phase="preview", processor=None,
                                  message="Camera connected · preview only. Detection is unavailable; see the error details.")

        def interrupt_all():
            for name, counter in counters.items():
                visit = counter.interrupt()
                if visit and name in active_ids:
                    store.update_visit(active_ids.pop(name), visit)
            store.break_coverage()

        try:
            import cv2
            camera = Camera(source, cv2).start()
            source = ""
            with self.lock:
                self.camera = camera
            deadline = time.monotonic() + 25
            sample = camera.latest()
            while (sample is None or time.time() - sample[1] > 10) and not self.stop_event.is_set():
                if getattr(camera, "status", "") == "Unavailable" or time.monotonic() > deadline:
                    raise getattr(camera, "failure_exception", None) or RuntimeError("No frames received from the camera within 25 seconds.")
                if self.stop_event.wait(.05):
                    break
                sample = camera.latest()
            if self.stop_event.is_set():
                return
            with self.lock:
                self.state.update(phase="loading_model", message="Camera connected. Loading the detector…",
                                  frame_shape=list(sample[2].shape[:2]))
            stage = "storage"
            store = Store(LOCAL / "data" / f"visits_{profile_stem(config['cat_label'])}.sqlite3")
            stage = "model"
            try:
                start_detector(config)
            except Exception as error:
                detector_failed("model", error)
            stage = "monitor"
            while not self.stop_event.wait(.025):
                if detector and detector.failure:
                    error = getattr(detector, "failure_exception", None) or RuntimeError(detector.failure)
                    detector_failed("detection", error)
                now, tick = time.time(), time.monotonic()
                sample = camera.latest()
                if sample is None or now - sample[1] > 10:
                    if not disconnected:
                        interrupt_all()
                    disconnected = True
                    with self.lock:
                        self.state.update(phase="connecting", message="Waiting for live camera video…",
                                          detections=[], tracks=[])
                    continue
                sequence, timestamp, frame = sample
                if sequence != last_sequence:
                    video_rate.record(tick)
                    last_sequence = sequence
                if disconnected:
                    interrupt_all()
                    disconnected = False
                prediction = detector.latest() if detector else None
                if prediction is not None and prediction.sequence != last_prediction:
                    last_prediction = prediction.sequence
                    if now - prediction.timestamp <= 10:
                        if last_prediction_time is not None and prediction.timestamp - last_prediction_time > 10:
                            store.break_coverage()
                            for counter in counters.values():
                                counter.observe(None, prediction.timestamp)
                                if counter.visit:
                                    counter.visit.status = "uncertain"
                        last_prediction_time = prediction.timestamp
                        height, width = prediction.image_shape[:2]
                        rectangles = pixel_zones(config["zones"], width, height)
                        cats = [box for box, _, label in prediction.detections if label in cat_ids]
                        occupied, ambiguous = (assign_station(cats[0], rectangles) if len(cats) == 1
                                               else (None, set(rectangles) if len(cats) > 1 else set()))
                        if counters:
                            store.mark_coverage(prediction.timestamp)
                        for name, counter in counters.items():
                            present = station_presence(counter, name, cats, rectangles, occupied, ambiguous)
                            for event, visit in counter.observe(present, prediction.timestamp):
                                if event == "start":
                                    active_ids[name] = store.start_visit(name, config["cat_label"], visit)
                                elif name in active_ids:
                                    store.update_visit(active_ids.pop(name), visit)
                            if counter.visit and name in active_ids:
                                store.update_visit(active_ids[name], counter.visit)
                        boxes = [{"box": [v / scale for v, scale in zip(box, (width, height, width, height))],
                                  "label": model.names[label], "score": score}
                                 for box, score, label in prediction.detections]
                        tracks = [{"id": t.track_id, "box": [v / scale for v, scale in
                                   zip(t.box, (width, height, width, height))], "score": t.confidence,
                                   "held": t.estimated, "last_seen": t.last_seen_at} for t in prediction.tracks]
                        with self.lock:
                            self.state.update(detections=boxes, tracks=tracks, ai_fps=prediction.fps,
                                              prediction_at=prediction.timestamp)
                if prediction is None or now - prediction.timestamp > 10:
                    store.break_coverage()
                    for counter in counters.values():
                        counter.observe(None, now)
                if tick - last_summary >= 1:
                    phase = "live" if detector else "preview"
                    message = ("Monitoring station visits." if counters else "Live preview · select station zones to count visits.") if detector else "Camera connected · preview only. Detection is unavailable; see the error details."
                    with self.lock:
                        self.state.update(summary=store.summary(now), phase=phase, frame_shape=list(frame.shape[:2]),
                                          message=self.state["notice"] or message,
                                          processor=("GPU 0" if device == "0" else device.upper()) if detector else None,
                                          video_fps=video_rate.value(tick))
                    last_summary = tick
                with self.lock:
                    collect = self.collecting and self.collection_count < 120
                    session = self.collection_session
                if collect and tick - last_capture >= 15:
                    # Collection is independent of cat detections: missed cats,
                    # dog misclassifications, and empty scenes are all reviewable.
                    thumbnail = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
                    changed = collection_previous is None or cv2.absdiff(thumbnail, collection_previous).mean() > 2
                    if changed or tick - last_capture >= 60:
                        ok, encoded = cv2.imencode(".jpg", frame)
                        if ok:
                            try:
                                self.library.add_image(session, encoded.tobytes(), frame.shape[1], frame.shape[0])
                                with self.lock:
                                    self.collection_count += 1
                                    if self.collection_count >= 120:
                                        self.collecting = False
                            except ValueError:
                                pass
                        collection_previous, last_capture = thumbnail, tick
                if tick - last_export >= 10:
                    store.export(LOCAL / "reports" / f"visits_{profile_stem(config['cat_label'])}.csv")
                    last_export = tick
        except Exception as error:
            row = report(stage, error)
            with self.lock:
                self.state.update(phase="error", message=row["message"],
                                  detections=[], tracks=[])
        finally:
            if store:
                try:
                    interrupt_all()
                    store.export(LOCAL / "reports" / f"visits_{profile_stem(config['cat_label'])}.csv")
                    with self.lock:
                        self.state["summary"] = store.summary(time.time())
                except Exception as error:
                    if self.state["error"] is None:
                        report("storage", error)
                finally:
                    try:
                        store.close()
                    except Exception:
                        pass
            for resource in (detector, camera):
                if resource:
                    try:
                        resource.stop()
                    except Exception as error:
                        if self.state["error"] is None:
                            report("cleanup", error)
            with self.lock:
                self.camera = None
                self.collecting = False
                self.collection_session = None
                self.state.update(ai_fps=0, video_fps=0)
                if self.state["phase"] != "error":
                    self.state.update(phase="offline", message="Camera disconnected.", detections=[], tracks=[])

    def close(self):
        self.training.cancel()
        self.disconnect()
