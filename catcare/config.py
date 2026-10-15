"""Nonsecret settings and predictable profile paths."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "local"
DEFAULTS = {"cat_label": "Cat", "model": "yolo26n.pt", "device": "auto",
            "confidence": 0.25, "image_size": 960, "crop": True, "zones": {}}


def load_settings():
    path = LOCAL / "settings.json"
    if not path.exists():
        return dict(DEFAULTS)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return {key: values.get(key, default) for key, default in DEFAULTS.items()}
    except (ValueError, OSError, AttributeError):
        return dict(DEFAULTS)


def validate_settings(values):
    label = values["cat_label"].strip()
    if not label or len(label) > 60 or "://" in label or any(ord(c) < 32 for c in label):
        raise ValueError("Use a short cat name, without URLs or control characters.")
    model = values["model"]
    if model not in ("yolo26n.pt", "yolo26s.pt"):
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
