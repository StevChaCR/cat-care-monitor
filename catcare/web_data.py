"""Private, profile-scoped images and manually reviewed detector labels."""
import hashlib
import json
import re
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

    def image_path(self, key):
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid image reference.")
        session, image = key.split("_")
        path = self._session(session) / "images" / f"{image}.jpg"
        if not path.is_file():
            raise ValueError("That image is no longer available.")
        return path

    def images(self):
        with self.lock:
            result = []
            for session in self.sessions():
                folder = self.root / session["id"]
                for meta in sorted((folder / "images").glob("*.json")):
                    row = json.loads(meta.read_text(encoding="utf-8"))
                    image = self.image_path(row["id"])
                    label = folder / "labels" / f"{image.stem}.txt"
                    boxes = []
                    if label.exists():
                        boxes = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                                 for _, x, y, w, h in read_labels(label)]
                    result.append({**row, "session_name": session["name"],
                                   "reviewed": label.exists(), "boxes": boxes})
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
