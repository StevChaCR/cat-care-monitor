"""Private, manually reviewed sequences for an independent activity model."""
import hashlib
import json
import math
import random
import re
import shutil
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

    @property
    def trash_root(self):
        folder = self.root / "trash"
        if folder.is_symlink():
            raise ValueError("The behavior trash must be a local folder inside this library.")
        return folder

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
                raise ValueError("This profile has reached the 600 active clip limit. Move unwanted clips to trash before collecting more.")
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

    def frame_path(self, key, index, trashed=False):
        if type(index) is not int or not 0 <= index < CLIP_FRAMES:
            raise ValueError("Invalid clip frame index.")
        folder = self.trashed_path(key) if trashed else self.clip_path(key)
        return folder / f"{index:02d}.jpg"

    def trashed_path(self, key):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid behavior clip reference.")
        folder = self.trash_root / key
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.trash_root.resolve()):
            raise ValueError("Invalid behavior trash reference.")
        if not (folder / "clip.json").is_file():
            raise ValueError("That clip is not in the trash.")
        return folder

    def trashed_clips(self):
        with self.lock:
            sessions = {row["id"]: row["name"] for row in self.sessions()}
            result = []
            for path in sorted(self.trash_root.glob("*/clip.json")):
                if KEY.fullmatch(path.parent.name):
                    folder = self.trashed_path(path.parent.name)
                    row = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
                    result.append({**row, "session_name": sessions.get(row["session_id"], "Original recording period")})
            return result

    @staticmethod
    def _clip_keys(keys):
        if not isinstance(keys, list) or not 1 <= len(keys) <= 600:
            raise ValueError("Choose between 1 and 600 behavior clips.")
        if any(not isinstance(key, str) or not KEY.fullmatch(key) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Invalid behavior clip selection.")
        return keys

    def move_to_trash(self, keys):
        with self.lock:
            keys = self._clip_keys(keys)
            # Validate the entire selection before moving any folders.
            sources = [self.clip_path(key) for key in keys]
            destinations = [self.trash_root / key for key in keys]
            if any(path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()) for path in sources):
                raise ValueError("Invalid behavior clip location.")
            if any(path.exists() for path in destinations):
                raise ValueError("A selected clip is already in the trash.")
            self.trash_root.mkdir(parents=True, exist_ok=True)
            for source, destination in zip(sources, destinations):
                source.replace(destination)
            return len(keys)

    def restore_clip(self, key):
        with self.lock:
            source = self.trashed_path(key)
            session, clip = key.split("_")
            destination = self.session_path(session) / clip
            rows = self.clips()
            if len(rows) >= 600:
                raise ValueError("The profile already has 600 active clips. Remove clips before restoring more.")
            if destination.exists():
                raise ValueError("That clip is already present in the library.")
            row = json.loads((source / "clip.json").read_text(encoding="utf-8"))
            if any(item["sha256"] == row["sha256"] for item in rows):
                raise ValueError("An identical clip is already present in the library.")
            source.replace(destination)

    def purge_clips(self, keys):
        with self.lock:
            sources = [self.trashed_path(key) for key in self._clip_keys(keys)]
            # Only explicitly selected, previously trashed clips can be erased.
            for source in sources:
                shutil.rmtree(source)
            return len(sources)

    def label_clip(self, key, label):
        if label not in (*ACTIVITIES, "exclude"):
            raise ValueError("Choose eating, drinking, other, or cannot tell.")
        with self.lock:
            path = self.clip_path(key) / "clip.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["label"] = label
            write_json(path, row)


def _random_behavior_split(session_ids, seed):
    session_ids = sorted(session_ids)
    random.Random(seed).shuffle(session_ids)
    held = max(1, round(len(session_ids) * .15))
    return {"test": session_ids[:held], "val": session_ids[held:2 * held],
            "train": session_ids[2 * held:]}


def _behavior_split_ids(groups, seed):
    """Keep whole periods and find class coverage before rejecting a random split."""
    original = _random_behavior_split(groups, seed)
    requirements = ((3, 3, 3, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1))
    roles = ("train", "val", "test")
    def features(items):
        return tuple(sum(row["label"] == label and (station is None or row["station"] == station) for row in items)
                     for label, station in (("eating", None), ("drinking", None), ("other", None),
                                            ("other", "food"), ("other", "water")))
    vectors = {key: features(items) for key, items in groups.items()}
    if all(all(sum(vectors[key][f] for key in original[role]) >= need
                   for f, need in enumerate(requirements[index])) for index, role in enumerate(roles)):
        return original
    support = tuple(sum(row[f] > 0 for row in vectors.values()) for f in range(5))
    if len(groups) < 3 or any(n < 3 for n in support):
        return original
    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda key: -sum(1 / support[f] for f, count in enumerate(vectors[key]) if count))
    suffix_counts = [(0,) * 5 for _ in range(len(order) + 1)]
    suffix_support = [(0,) * 5 for _ in range(len(order) + 1)]
    for at in range(len(order) - 1, -1, -1):
        vector = vectors[order[at]]
        suffix_counts[at] = tuple(suffix_counts[at + 1][f] + vector[f] for f in range(5))
        suffix_support[at] = tuple(suffix_support[at + 1][f] + (vector[f] > 0) for f in range(5))
    assignments = {role: [] for role in roles}
    failed = set()
    visited = 0
    def find(at, deficits):
        nonlocal visited
        visited += 1
        if visited > 50000:
            return None  # Bound preparation time; unresolved splits remain blocked.
        if not any(any(row) for row in deficits):
            result = {role: list(values) for role, values in assignments.items()}
            result["train"].extend(order[at:])
            return result
        if at == len(order):
            return None
        state = (at, deficits)
        if state in failed:
            return None
        for f in range(5):
            if (sum(row[f] for row in deficits) > suffix_counts[at][f] or
                    sum(row[f] > 0 for row in deficits) > suffix_support[at][f]):
                return None
        key, vector = order[at], vectors[order[at]]
        gains = [sum(min(vector[f], need[f]) / support[f] for f in range(5)) for need in deficits]
        choices = sorted(range(3), key=lambda index: (-gains[index], index))
        for index in choices:
            if index == 2 and deficits[1] == deficits[2]:
                continue  # Validation and test have identical coverage requirements.
            if index != 0 and gains[index] == 0:
                continue
            changed = tuple(max(0, need - count) for need, count in zip(deficits[index], vector))
            updated = tuple(changed if n == index else row for n, row in enumerate(deficits))
            assignments[roles[index]].append(key)
            found = find(at + 1, updated)
            assignments[roles[index]].pop()
            if found is not None:
                return found
        failed.add(state)
        return None
    return find(0, requirements) or original


def behavior_readiness(source, seed=42, rows=None, sessions=None):
    """Explain the current session split without loading every JPEG or training."""
    library = ClipLibrary(source)
    rows = library.clips() if rows is None else rows
    sessions = library.sessions() if sessions is None else sessions
    names = {row["id"]: row["name"] for row in sessions}
    counts = {activity: sum(row["label"] == activity for row in rows) for activity in ACTIVITIES}
    unreviewed = sum(row["label"] is None for row in rows)
    excluded = sum(row["label"] == "exclude" for row in rows)
    issues = []
    if not rows:
        issues.append("Collect and review behavior clips first. Cat detection photos are a separate dataset.")
    if unreviewed:
        issues.append(f"Review every behavior clip: {unreviewed} still need a label. Use Cannot tell or trash for unusable clips.")
    if any(row["label"] not in (*ACTIVITIES, "exclude", None) for row in rows):
        issues.append("An activity label is invalid. Review that clip before training.")
    for activity, count in counts.items():
        if count == 0:
            detail = " Include nearby resting, sniffing, passing cats, and empty scenes at both stations." if activity == "other" else ""
            issues.append(f"No reviewed {activity.title()} clips are available.{detail}")
    groups = {}
    for row in rows:
        if row["label"] in ACTIVITIES:
            groups.setdefault(row["session_id"], []).append(row)
    if len(groups) < 3:
        issues.append(f"Use at least three independent behavior recording sessions from different real periods; found {len(groups)} with usable labels. "
                      "A period is one genuine recording occasion, such as Monday morning or Tuesday evening. Empty session names do not count.")
    overlaps = []
    for index, a in enumerate(sorted(groups)):
        for b in sorted(groups)[index + 1:]:
            a_start, a_end = min(row["timestamps"][0] for row in groups[a]), max(row["timestamps"][-1] for row in groups[a])
            b_start, b_end = min(row["timestamps"][0] for row in groups[b]), max(row["timestamps"][-1] for row in groups[b])
            if min(a_end, b_end) >= max(a_start, b_start):
                overlaps.append([names.get(a, "Recording period"), names.get(b, "Recording period")])
    if overlaps:
        first = overlaps[0]
        issues.append(f"Behavior sessions overlap in time: {first[0]} and {first[1]}. Keep simultaneous station clips in one real recording period.")
    split_ids = _behavior_split_ids(groups, seed)
    roles = {session: split for split, selected in split_ids.items() for session in selected}
    splits = {}
    for split in ("train", "val", "test"):
        selected = split_ids[split]
        items = [row for session in selected for row in groups[session]]
        minimum = 3 if split == "train" else 1
        split_counts = {activity: sum(row["label"] == activity for row in items) for activity in ACTIVITIES}
        minimums = {activity: max(2, minimum) if activity == "other" else minimum for activity in ACTIVITIES}
        negatives = {station: sum(row["label"] == "other" and row["station"] == station for row in items)
                     for station in ("food", "water")}
        missing = [f"{activity.title()}: {count}/{minimums[activity]}" for activity, count in split_counts.items() if count < minimums[activity]]
        missing += [f"Other at {station}: 0/1" for station, count in negatives.items() if count == 0]
        splits[split] = {"periods": [names.get(session, "Recording period") for session in selected],
                         "clips": len(items), "counts": split_counts, "minimum_per_class": minimum, "minimums": minimums,
                         "other_by_station": negatives, "missing": missing}
        if len(groups) >= 3 and missing:
            issues.append(f"{split.title()} split is missing " + "; ".join(missing) + ". Add those reviewed examples in its listed recording periods or collect additional independent periods.")
    periods = []
    for session in sessions:
        items = [row for row in rows if row["session_id"] == session["id"]]
        periods.append({"id": session["id"], "name": session["name"], "clips": len(items),
                        "counts": {activity: sum(row["label"] == activity for row in items) for activity in ACTIVITIES},
                        "unreviewed": sum(row["label"] is None for row in items),
                        "excluded": sum(row["label"] == "exclude" for row in items),
                        "split": roles.get(session["id"]),
                        "start": min((row["timestamps"][0] for row in items), default=None),
                        "end": max((row["timestamps"][-1] for row in items), default=None)})
    return {"ready": not issues, "integrity_checked": False,
            "split_strategy": "session_shuffle" if split_ids == _random_behavior_split(groups, seed) else "class_coverage_by_period",
            "clips": len(rows), "counts": counts,
            "unreviewed": unreviewed, "excluded": excluded, "usable_periods": len(groups),
            "periods": periods, "splits": splits, "issues": issues}


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
    split_ids = _behavior_split_ids(groups, seed)
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
