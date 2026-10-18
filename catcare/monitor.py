"""Local video inference and three-station monitoring."""
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from .camera import Camera
from .config import LOCAL, profile_stem, save_settings
from .core import STATIONS, VisitCounter, assign_station, pixel_zones, station_presence, validate_zones
from .detector import LatestDetector, processing_crop
from .display import FrameRate, fit_frame, normalized_roi, screen_bounds, status_view
from .storage import Store
from .models import resolve_detector

WINDOW = "Cat Care Monitor | Q quit | S save frame | D detection labels"


def read_key(cv2, delay=1):
    key = cv2.waitKey(delay) & 0xff
    try:
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
    except cv2.error:
        return ord("q")
    return key


def initialize_model(config):
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(LOCAL / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    from ultralytics import YOLO, settings
    import torch
    settings.update({"sync": False, "weights_dir": str(LOCAL / "models"),
                     "runs_dir": str(LOCAL / "training"), "datasets_dir": str(LOCAL / "dataset")})
    device = config["device"]
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.ones(1, device="cuda:0").add_(1).cpu()
                device = "0"
            except RuntimeError:
                pass
    path = resolve_detector(config["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Loading the detector. The first run may download pretrained weights.")
    model = YOLO(str(path))
    cat_ids = [index for index, name in model.names.items() if name.lower() == "cat"]
    if not cat_ids:
        raise ValueError("The selected model must include a class named 'cat'.")
    print("Processor:", "GPU 0" if device == "0" else device)
    return model, cat_ids, device


def select_zones(frame, cv2):
    # Always fit the COMPLETE camera image, independently of inference crops.
    display = fit_frame(frame, cv2, screen_bounds())
    zones = {}
    for station in STATIONS:
        title = f"Select {station.replace('_', ' ')} | ENTER confirms | C skips"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(title, 40, 40)
        x, y, w, h = cv2.selectROI(title, display, False, False)
        cv2.destroyWindow(title)
        if w and h:
            zones[station] = normalized_roi((x, y, w, h), display.shape)
    validate_zones(zones)
    return zones


def run_monitor(config, source, preview=False):
    model, cat_ids, device = initialize_model(config)
    import cv2
    import numpy as np
    display_size = screen_bounds()
    camera = Camera(source, cv2).start()
    detector = None
    store = None
    counters = {}
    active_ids = {}
    csv_path = LOCAL / "reports" / f"visits_{profile_stem(config['cat_label'])}.csv"
    session = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S_%f")
    last_sequence = None
    last_frame_time = None
    last_prediction_sequence = None
    last_prediction_time = None
    last_summary = 0.0
    last_export = 0.0
    video_rate = FrameRate()
    summary = {"counts": {}, "durations": {}, "uncertain": {}, "observed_seconds": 0}
    detection_labels = preview
    disconnected = False

    def interrupt_all():
        for station, counter in counters.items():
            visit = counter.interrupt()
            if visit and store is not None and station in active_ids:
                store.update_visit(active_ids.pop(station), visit)
        if store is not None:
            store.break_coverage()

    def pause_all(timestamp, uncertain=False):
        for station, counter in counters.items():
            previous_status = counter.visit.status if counter.visit else None
            counter.observe(None, timestamp)
            if uncertain and counter.visit:
                counter.visit.status = "uncertain"
            if (counter.visit and store is not None and station in active_ids and
                    previous_status != counter.visit.status):
                store.update_visit(active_ids[station], counter.visit)
        if store is not None:
            store.break_coverage()

    try:
        print("Connecting to the local camera...")
        deadline = time.monotonic() + 20
        sample = None
        while sample is None and time.monotonic() < deadline:
            sample = camera.latest()
            time.sleep(0.05)
        if sample is None:
            raise ValueError("No video received. Test the same complete URL in VLC and check the local network.")
        if not config["zones"]:
            config["zones"] = select_zones(sample[2], cv2)
            save_settings(config)
        validate_zones(config["zones"])
        counters = {name: VisitCounter() for name in config["zones"]}
        if not preview:
            store = Store(LOCAL / "data" / f"visits_{profile_stem(config['cat_label'])}.sqlite3")
        print("Preview only." if preview else "Wait 8 seconds with empty zones before visit counting starts.")
        print("S saves a review frame locally. D shows other detected classes. Q stops.")
        detector = LatestDetector(camera, model, config, cat_ids, device, detection_labels).start()
        # The canvas already fits the desktop: do not shrink its status text again.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW, 40, 40)
        while True:
            tick = time.monotonic()
            now = time.time()
            if detector.failure:
                raise ValueError(detector.failure)
            sample = camera.latest()
            if sample is None or now - sample[1] > 10:
                if not disconnected:
                    interrupt_all()
                    print("Camera unavailable. Active visits marked interrupted; waiting for live video.")
                    disconnected = True
                waiting = np.zeros((1, 1, 3), dtype=np.uint8)
                cv2.imshow(WINDOW, status_view(waiting,
                    ["Waiting for live camera video... Q exits"], cv2, display_size))
                if read_key(cv2, 50) in (ord("q"), ord("Q")):
                    break
                continue
            sequence, timestamp, frame = sample
            if sequence != last_sequence:
                video_rate.record(tick)
            last_sequence = sequence
            if disconnected or (last_frame_time is not None and timestamp - last_frame_time > 10):
                interrupt_all()
            disconnected = False
            last_frame_time = timestamp
            height, width = frame.shape[:2]
            rectangles = pixel_zones(config["zones"], width, height)
            x1, y1, x2, y2 = processing_crop(rectangles, frame.shape, config["crop"])
            # Training needs the original camera image, without display overlays.
            raw_frame = frame.copy()
            prediction = detector.latest()
            cats = [] if prediction is None else [box for box, _, class_id in prediction.detections if class_id in cat_ids]
            occupied, ambiguous = None, set()
            predicted_rectangles = rectangles if prediction is None else pixel_zones(
                config["zones"], prediction.image_shape[1], prediction.image_shape[0])
            if len(cats) == 1:
                occupied, ambiguous = assign_station(cats[0], predicted_rectangles)
            elif len(cats) > 1:
                ambiguous = set(predicted_rectangles)
            new_prediction = prediction is not None and prediction.sequence != last_prediction_sequence
            if new_prediction:
                last_prediction_sequence = prediction.sequence
            if new_prediction and store is not None and now - prediction.timestamp <= 10:
                if last_prediction_time is not None and prediction.timestamp - last_prediction_time > 10:
                    pause_all(prediction.timestamp, uncertain=True)
                last_prediction_time = prediction.timestamp
                # Repainting the same boxes never advances visit time or coverage.
                store.mark_coverage(prediction.timestamp)
                for station, counter in counters.items():
                    present = station_presence(counter, station, cats, predicted_rectangles, occupied, ambiguous)
                    for event, visit in counter.observe(present, prediction.timestamp):
                        if event == "start":
                            active_ids[station] = store.start_visit(station, config["cat_label"], visit)
                        elif station in active_ids:
                            store.update_visit(active_ids.pop(station), visit)
                    if counter.visit and station in active_ids:
                        store.update_visit(active_ids[station], counter.visit)
            if store is not None and tick - last_summary >= 1:
                summary = store.summary(now)
                last_summary = tick
            if prediction is not None and now - prediction.timestamp <= 2:
                scale_x = width / prediction.image_shape[1]
                scale_y = height / prediction.image_shape[0]
                for box, score, class_id in prediction.detections:
                    if class_id in cat_ids and prediction.tracks:
                        continue  # Draw the persistent track ID once, below.
                    left, top, right, bottom = box
                    box = (int(left * scale_x), int(top * scale_y), int(right * scale_x), int(bottom * scale_y))
                    color = (0, 220, 0) if class_id in cat_ids else (0, 160, 255)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{model.names[class_id]} {score:.2f}", (box[0], max(20, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            held_tracks = []
            if prediction is not None:
                scale_x = width / prediction.image_shape[1]
                scale_y = height / prediction.image_shape[0]
                for track in prediction.tracks:
                    age = max(0, now - track.last_seen_at)
                    if age > 5:
                        continue
                    held = track.estimated or now - prediction.timestamp > 2
                    box = tuple(int(value * scale) for value, scale in
                                zip(track.box, (scale_x, scale_y, scale_x, scale_y)))
                    color = (0, 170, 255) if held else (0, 220, 0)
                    label = f"Cat #{track.track_id} | held" if held else f"Cat #{track.track_id} {track.confidence:.2f}"
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, label, (box[0], max(20, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    if held:
                        held_tracks.append(track)
            if prediction is None or now - prediction.timestamp > 10:
                pause_all(now)
            for station, rect in rectangles.items():
                a, b, c, d = rect
                color = (0, 160, 255) if station in ambiguous else (255, 170, 0)
                cv2.rectangle(frame, (a, b), (c, d), color, 2)
                cv2.putText(frame, station.replace("_", " "), (a, max(18, b - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            view = frame[y1:y2, x1:x2]
            mode = "PREVIEW - no visit logging" if preview else "LAST 24 HOURS"
            lines = [f"{config['cat_label'][:24]} | {mode} | Cats detected: {len(cats)}"]
            ai_fps = 0 if prediction is None or now - prediction.timestamp > 10 else prediction.fps
            processor = "GPU 0" if device == "0" else device.upper()
            lines.append(f"Video: {video_rate.value(tick):.1f} FPS | AI: {ai_fps:.1f} FPS | {processor}")
            for station in STATIONS:
                if station not in rectangles:
                    lines.append(f"{station.replace('_', ' ').title()}: not configured")
                else:
                    count = summary["counts"].get(station, 0)
                    duration = summary["durations"].get(station, 0) / 60
                    incomplete = summary["uncertain"].get(station, 0)
                    lines.append(f"{station.replace('_', ' ').title()}: {count} visits | {duration:.1f} observed min | {incomplete} incomplete")
            lines.append(f"Observed coverage: {summary['observed_seconds'] / 3600:.2f} / 24 hours")
            if held_tracks:
                lines.append("Tracking: holding the last cat box; visit waits for an observed exit")
            elif not cats and any(counter.visit for counter in counters.values()):
                lines.append("Cat not observed: current visit paused; no new entry will be counted")
            else:
                lines.append("AMBIGUOUS: adjust zones or keep one cat in view" if ambiguous else
                             "Visits are station presence, not confirmed eating, drinking, or elimination")
            cv2.imshow(WINDOW, status_view(view, lines, cv2, display_size))
            delay = max(1, min(33, round((tick + 1 / 30 - time.monotonic()) * 1000)))
            key = read_key(cv2, delay)
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D")):
                detection_labels = not detection_labels
                detector.set_show_all(detection_labels)
            if key in (ord("s"), ord("S")):
                directory = LOCAL / "captures" / session / "images"
                directory.mkdir(parents=True, exist_ok=True)
                filename = datetime.now(timezone.utc).strftime("frame_%H%M%S_%f.jpg")
                if cv2.imwrite(str(directory / filename), raw_frame):
                    print("Review frame saved:", directory / filename)
            if store is not None and time.monotonic() - last_export >= 10:
                if not store.export(csv_path):
                    print("Close the CSV in Excel to refresh it. SQLite records continue normally.")
                last_export = time.monotonic()
    finally:
        if detector is not None:
            detector.stop()
        camera.stop()
        cv2.destroyAllWindows()
        if store is not None:
            try:
                interrupt_all()
                exported = store.export(csv_path)
            finally:
                store.close()
            print("Visit records saved to SQLite.")
            if exported:
                print("CSV:", csv_path)
            else:
                print("CSV refresh pending. Close Excel and run the summary command.")
