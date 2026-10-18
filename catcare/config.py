"""Nonsecret settings and predictable profile paths."""
import hashlib
import json
import math
import re
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def data_directory():
    if not getattr(sys, "frozen", False):
        return ROOT / "local"  # Preserve existing source installations unchanged.
    if sys.platform == "win32":
        parent = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        parent = Path.home() / "Library" / "Application Support"
    else:
        parent = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return parent / "WatchMyCat"


LOCAL = data_directory()
DEFAULTS = {"cat_label": "Cat", "model": "yolo26n.pt", "device": "auto",
            "confidence": 0.25, "image_size": 960, "crop": True, "zones": {},
            "behavior_model": "bundled"}


def load_settings():
    from .models import prefer_shared_detector
    path = LOCAL / "settings.json"
    values = dict(DEFAULTS)
    try:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            values = {key: saved.get(key, default) for key, default in DEFAULTS.items()}
    except (ValueError, OSError, AttributeError):
        pass
    values["model"] = prefer_shared_detector(values["model"])
    return values


def validate_settings(values):
    label = values["cat_label"].strip()
    if not label or len(label) > 60 or "://" in label or any(ord(c) < 32 for c in label):
        raise ValueError("Use a short cat name, without URLs or control characters.")
    model = values["model"]
    if model == "bundled":
        from .models import bundled_detector
        bundled_detector()
    elif model not in ("yolo26n.pt", "yolo26s.pt"):
        path = Path(model)
        if "://" in model or path.suffix.lower() != ".pt" or not path.is_file():
            raise ValueError("Choose a pretrained model or an existing local .pt file.")
    try:
        confidence = float(values["confidence"])
        size = int(values["image_size"])
    except (TypeError, ValueError):
        raise ValueError("Confidence and image size must be numbers.") from None
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Confidence must be greater than 0 and at most 1.")
    if not 32 <= size <= 2048:
        raise ValueError("Image size must be between 32 and 2048.")
    if values["device"] not in ("auto", "cpu", "0"):
        raise ValueError("Choose auto, cpu, or GPU 0.")
    activity_model = values.get("behavior_model", "bundled")
    if activity_model not in ("", "bundled") and not (isinstance(activity_model, str) and re.fullmatch(r"[a-f0-9]{32}", activity_model)):
        raise ValueError("Choose the included activity model or a completed local training run.")
    return {**values, "cat_label": label, "confidence": confidence, "image_size": size}


def save_settings(values):
    LOCAL.mkdir(parents=True, exist_ok=True)
    # This allowlist deliberately has no URL, username, password, or endpoint key.
    allowed = {key: values[key] for key in DEFAULTS}
    temporary = LOCAL / "settings.tmp"
    temporary.write_text(json.dumps(allowed, indent=2), encoding="utf-8")
    temporary.replace(LOCAL / "settings.json")


def profile_stem(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]


def activity_model_path(profile, selected):
    if selected == "bundled":
        path = ROOT / "assets" / "behavior" / "model.json"
    elif isinstance(selected, str) and re.fullmatch(r"[a-f0-9]{32}", selected):
        path = Path(profile) / "behavior" / "training" / selected / "model.json"
    else:
        return None
    return path if path.is_file() else None
