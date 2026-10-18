"""Private, manually reviewed sequences for an independent activity model."""
import hashlib
import json
import math
import random
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .behavior import ACTIVITIES, CLIP_FRAMES

ID = re.compile(r"[a-f0-9]{32}\Z")
KEY = re.compile(r"[a-f0-9]{32}_[a-f0-9]{32}\Z")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class ClipLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    def sessions(self):
        with self.lock:
            return [json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(self.root.glob("*/session.json")) if ID.fullmatch(path.parent.name)]

    def create_session(self, name):
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError("Give this real recording period a short name.")
        with self.lock:
            row = {"id": uuid.uuid4().hex, "name": name,
                   "created_at": datetime.now(timezone.utc).isoformat()}
            write_json(self.root / row["id"] / "session.json", row)
            return row

    def session_path(self, session_id):
        if not isinstance(session_id, str) or not ID.fullmatch(session_id):
            raise ValueError("Choose a behavior recording session first.")
        folder = self.root / session_id
        if not (folder / "session.json").is_file():
            raise ValueError("That behavior session does not exist.")
        return folder

    def clip_path(self, key):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid behavior clip reference.")
        session, clip = key.split("_")
        folder = self.session_path(session) / clip
        if not (folder / "clip.json").is_file():
            raise ValueError("That behavior clip does not exist.")
        return folder

    def clips(self):
        with self.lock:
            result = []
            for session in self.sessions():
                for path in sorted((self.root / session["id"]).glob("*/clip.json")):
                    if ID.fullmatch(path.parent.name):
                        row = json.loads(path.read_text(encoding="utf-8"))
                        result.append({**row, "session_name": session["name"]})
            return result

    def add_clip(self, session_id, station, frames, timestamps, roi):
        if station not in ("food", "water") or len(frames) != CLIP_FRAMES or len(timestamps) != CLIP_FRAMES:
            raise ValueError("A behavior clip needs 16 consecutive station frames.")
        if any(not isinstance(t, (float, int)) or not math.isfinite(t) for t in timestamps):
            raise ValueError("Invalid clip timestamps.")
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        if any(not .18 <= gap <= .75 for gap in gaps) or timestamps[-1] - timestamps[0] > 6:
            raise ValueError("Camera frames are too sparse or discontinuous for this clip.")
        if any(not isinstance(frame, bytes) or not 0 < len(frame) <= 2_000_000 for frame in frames):
            raise ValueError("Invalid encoded clip frames.")
        digest = hashlib.sha256(b"".join(frames)).hexdigest()
        with self.lock:
            folder = self.session_path(session_id)
            rows = self.clips()
            if len(rows) >= 600:
                raise ValueError("This profile has reached the 600-clip limit. Export or review its data before collecting more.")
            if any(row["sha256"] == digest for row in rows):
                raise ValueError("This exact clip has already been collected.")
            clip_id = uuid.uuid4().hex
            destination = folder / clip_id
            destination.mkdir()
            for index, frame in enumerate(frames):
                (destination / f"{index:02d}.jpg").write_bytes(frame)
            row = {"id": f"{session_id}_{clip_id}", "session_id": session_id,
                   "station": station, "frames": CLIP_FRAMES, "timestamps": list(timestamps),
                   "duration": timestamps[-1] - timestamps[0], "roi": list(roi),
                   "sha256": digest, "label": None}
            write_json(destination / "clip.json", row)
            return row

    def frame_path(self, key, index):
        if type(index) is not int or not 0 <= index < CLIP_FRAMES:
            raise ValueError("Invalid clip frame index.")
        return self.clip_path(key) / f"{index:02d}.jpg"

    def label_clip(self, key, label):
        if label not in (*ACTIVITIES, "exclude"):
            raise ValueError("Choose eating, drinking, other, or cannot tell.")
        with self.lock:
            path = self.clip_path(key) / "clip.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["label"] = label
            write_json(path, row)


def prepare_behavior_dataset(source, output, seed=42):
    """Keep whole recording periods together, with all classes in every split."""
    library = ClipLibrary(source)
    rows = library.clips()
    if not rows:
        raise ValueError("Collect and review behavior clips first. Cat detection photos are a separate dataset.")
    if any(row["label"] is None for row in rows):
        raise ValueError("Review every behavior clip, or mark unclear clips as Cannot tell.")
    groups, hashes = {}, {}
    for row in rows:
        if row["label"] == "exclude":
            continue
        if row["label"] not in ACTIVITIES:
            raise ValueError("An activity label is invalid.")
        # Hash actual stored frames; do not trust mutable metadata for splitting.
        digest = hashlib.sha256(b"".join(library.frame_path(row["id"], index).read_bytes()
                                        for index in range(CLIP_FRAMES))).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate behavior clips cannot be used as independent examples.")
        hashes[digest] = row["session_id"]
        groups.setdefault(row["session_id"], []).append(row)
    if len(groups) < 3:
        raise ValueError("Use at least three independent behavior recording sessions from different real periods.")
    session_ids = sorted(groups)
    random.Random(seed).shuffle(session_ids)
    held = max(1, round(len(session_ids) * .15))
    split_ids = {"test": session_ids[:held], "val": session_ids[held:2 * held],
                 "train": session_ids[2 * held:]}
    # Split overlapping source time ranges together rather than pretending a
    # reconnection or session-name change made the same footage independent.
    for a_index, a in enumerate(session_ids):
        a_start = min(row["timestamps"][0] for row in groups[a])
        a_end = max(row["timestamps"][-1] for row in groups[a])
        for b in session_ids[a_index + 1:]:
            b_start = min(row["timestamps"][0] for row in groups[b])
            b_end = max(row["timestamps"][-1] for row in groups[b])
            if min(a_end, b_end) >= max(a_start, b_start):
                raise ValueError("Behavior sessions overlap in time. Keep one real recording period in one session.")
    result = {"classes": list(ACTIVITIES), "splits": {}, "source": str(Path(source).resolve())}
    for split, selected in split_ids.items():
        items = [row for session in selected for row in groups[session]]
        counts = {label: sum(row["label"] == label for row in items) for label in ACTIVITIES}
        minimum = 3 if split == "train" else 1
        if any(count < minimum for count in counts.values()):
            raise ValueError(f"The {split} split needs at least {minimum} reviewed clip(s) of each activity: eating, drinking, and other. "
                             "Collect all three activities in independent recording periods.")
        negatives = {row["station"] for row in items if row["label"] == "other"}
        if negatives != {"food", "water"}:
            raise ValueError(f"The {split} split needs Other examples at both food and water stations.")
        result["splits"][split] = {"sessions": selected, "clips": items, "counts": counts}
    write_json(Path(output), result)
    return result
