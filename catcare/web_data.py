"""Private, profile-scoped images and manually reviewed detector labels."""
import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .dataset import read_labels

KEY = re.compile(r"[a-f0-9]{32}_[a-f0-9]{32}\Z")
SESSION = re.compile(r"[a-f0-9]{32}\Z")


class ImageLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    @property
    def trash_root(self):
        path = self.root / "trash"
        if path.is_symlink():
            raise ValueError("The image trash must be a local folder inside this library.")
        return path

    def sessions(self):
        with self.lock:
            result = []
            for path in sorted(self.root.glob("*/session.json")):
                if SESSION.fullmatch(path.parent.name):
                    result.append(json.loads(path.read_text(encoding="utf-8")))
            return result

    def create_session(self, name):
        name = str(name).strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            raise ValueError("Give this recording session a short name.")
        with self.lock:
            session_id = uuid.uuid4().hex
            folder = self.root / session_id
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir()
            row = {"id": session_id, "name": name,
                   "created_at": datetime.now(timezone.utc).isoformat()}
            (folder / "session.json").write_text(json.dumps(row), encoding="utf-8")
            return row

    def _session(self, session_id):
        if not isinstance(session_id, str) or not SESSION.fullmatch(session_id):
            raise ValueError("Choose a recording session first.")
        folder = self.root / session_id
        if not (folder / "session.json").is_file():
            raise ValueError("That recording session does not exist.")
        return folder

    def add_image(self, session_id, jpeg, width, height):
        """Accept a decoded and re-encoded JPEG; original EXIF is not retained."""
        with self.lock:
            folder = self._session(session_id)
            digest = hashlib.sha256(jpeg).hexdigest()
            for existing in self.images():
                if existing["sha256"] == digest:
                    raise ValueError("This image is already in this cat's library.")
            image_id = uuid.uuid4().hex
            (folder / "images" / f"{image_id}.jpg").write_bytes(jpeg)
            row = {"id": f"{session_id}_{image_id}", "session_id": session_id,
                   "width": width, "height": height, "sha256": digest}
            (folder / "images" / f"{image_id}.json").write_text(json.dumps(row), encoding="utf-8")
            return row

    def image_path(self, key, trashed=False):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid image reference.")
        if trashed:
            path = self._trashed_path(key) / "image.jpg"
            if not path.is_file():
                raise ValueError("That trashed image is no longer available.")
            return path
        session, image = key.split("_")
        path = self._session(session) / "images" / f"{image}.jpg"
        if not path.is_file():
            raise ValueError("That image is no longer available.")
        return path

    def _trashed_path(self, key):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid image reference.")
        folder = self.trash_root / key
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.trash_root.resolve()):
            raise ValueError("Invalid image trash reference.")
        if not (folder / "image.json").is_file():
            raise ValueError("That image is not in the trash.")
        return folder

    @staticmethod
    def _boxes(label):
        return [[x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                for _, x, y, w, h in read_labels(label)] if label.is_file() else []

    def trashed_images(self):
        with self.lock:
            sessions = {row["id"]: row["name"] for row in self.sessions()}
            result = []
            for meta in sorted(self.trash_root.glob("*/image.json")):
                if KEY.fullmatch(meta.parent.name):
                    folder = self._trashed_path(meta.parent.name)
                    row = json.loads((folder / "image.json").read_text(encoding="utf-8"))
                    label = folder / "label.txt"
                    result.append({**row, "session_name": sessions.get(row["session_id"], "Original recording session"),
                                   "reviewed": label.is_file(), "boxes": self._boxes(label)})
            return result

    @staticmethod
    def _keys(keys):
        if (not isinstance(keys, list) or not 1 <= len(keys) <= 600 or
                any(not isinstance(key, str) or not KEY.fullmatch(key) for key in keys) or len(keys) != len(set(keys))):
            raise ValueError("Choose between 1 and 600 distinct image references.")
        return keys

    def _safe_file(self, path):
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Invalid private image location.")
        return path

    @staticmethod
    def _move_files(pairs):
        moved = []
        try:
            for source, destination in pairs:
                source.replace(destination)
                moved.append((source, destination))
        except OSError:
            for source, destination in reversed(moved):
                destination.replace(source)
            raise

    def move_to_trash(self, keys):
        with self.lock:
            selected = []
            for key in self._keys(keys):
                image = self._safe_file(self.image_path(key))
                meta = self._safe_file(image.with_suffix(".json"))
                label = self._safe_file(image.parent.parent / "labels" / f"{image.stem}.txt")
                if not meta.is_file():
                    raise ValueError("The selected image metadata is missing.")
                destination = self.trash_root / key
                if destination.exists():
                    raise ValueError("A selected image is already in the trash.")
                pairs = [(image, destination / "image.jpg"), (meta, destination / "image.json")]
                if label.is_file():
                    pairs.append((label, destination / "label.txt"))
                selected.append((destination, pairs))
            for destination, pairs in selected:
                destination.mkdir(parents=True)
                try:
                    self._move_files(pairs)
                except OSError:
                    if not any(destination.iterdir()):
                        destination.rmdir()
                    raise
            return len(selected)

    def restore_image(self, key):
        with self.lock:
            source = self._trashed_path(key)
            image = self._safe_file(self.image_path(key, trashed=True))
            meta = self._safe_file(source / "image.json")
            label = self._safe_file(source / "label.txt")
            row = json.loads(meta.read_text(encoding="utf-8"))
            if any(item["sha256"] == row["sha256"] for item in self.images()):
                raise ValueError("An identical image is already present in the active library.")
            session, image_id = key.split("_")
            folder = self._session(session)
            targets = [self._safe_file(folder / "images" / f"{image_id}.jpg"),
                       self._safe_file(folder / "images" / f"{image_id}.json"),
                       self._safe_file(folder / "labels" / f"{image_id}.txt")]
            if any(path.exists() for path in targets):
                raise ValueError("That image or its label is already in the library.")
            pairs = [(image, targets[0]), (meta, targets[1])]
            if label.is_file():
                pairs.append((label, targets[2]))
            self._move_files(pairs)
            if not any(source.iterdir()):
                source.rmdir()

    def purge_images(self, keys):
        with self.lock:
            folders = [self._trashed_path(key) for key in self._keys(keys)]
            for folder in folders:
                shutil.rmtree(folder)
            return len(folders)

    def images(self):
        with self.lock:
            result = []
            for session in self.sessions():
                folder = self.root / session["id"]
                for meta in sorted((folder / "images").glob("*.json")):
                    row = json.loads(meta.read_text(encoding="utf-8"))
                    image = self.image_path(row["id"])
                    label = folder / "labels" / f"{image.stem}.txt"
                    result.append({**row, "session_name": session["name"],
                                   "reviewed": label.exists(), "boxes": self._boxes(label)})
            return result

    def save_label(self, key, boxes, negative=False):
        with self.lock:
            image = self.image_path(key)
            if not isinstance(boxes, list) or len(boxes) > 20:
                raise ValueError("Draw boxes around the visible cats.")
            if not boxes and negative is not True:
                raise ValueError("Draw a cat box, or explicitly mark an empty scene.")
            if boxes and negative:
                raise ValueError("An empty scene cannot have cat boxes.")
            lines = []
            for box in boxes:
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError("Invalid cat box.")
                left, top, right, bottom = map(float, box)
                lines.append(f"0 {(left+right)/2:.8f} {(top+bottom)/2:.8f} "
                             f"{right-left:.8f} {bottom-top:.8f}\n")
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            temporary = label.with_suffix(".tmp")
            temporary.write_text("".join(lines), encoding="utf-8")
            try:
                read_labels(temporary)
                temporary.replace(label)
            finally:
                temporary.unlink(missing_ok=True)
