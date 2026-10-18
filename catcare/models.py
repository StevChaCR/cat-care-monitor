"""Resolve the one selected detector and verify distributable model resources."""
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

STARTER_MODELS = ("yolo26n.pt", "yolo26s.pt")


def release_directory():
    from .config import ROOT
    return ROOT / "assets" / "detector"


def prefer_shared_detector(selected):
    if selected in STARTER_MODELS and (release_directory() / "model.json").is_file():
        return "bundled"
    return selected


@lru_cache(maxsize=8)
def _verified_file(path, size, modified, expected):
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("The included detector checksum does not match. Reinstall the complete release.")
    return Path(path)


def bundled_detector(directory=None):
    folder = Path(directory) if directory is not None else release_directory()
    manifest = folder / "model.json"
    if not manifest.is_file() or manifest.stat().st_size > 65536:
        raise ValueError("No shared detector is included. Bundle a reviewed detector before distributing the app.")
    try:
        row = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("The included detector manifest could not be read.") from None
    if (not isinstance(row, dict) or row.get("schema_version") != 1 or
            row.get("file") != "model.pt" or row.get("task") != "detect" or row.get("classes") != ["cat"] or
            not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
        raise ValueError("The included detector manifest is invalid.")
    weights = folder / "model.pt"
    if not weights.is_file() or not 0 < weights.stat().st_size <= 200_000_000:
        raise ValueError("The included detector weights are missing or exceed the size limit.")
    stat = weights.stat()
    return _verified_file(str(weights.resolve()), stat.st_size, stat.st_mtime_ns, row["sha256"])


def resolve_detector(selected):
    if selected == "bundled":
        return bundled_detector()
    if selected in STARTER_MODELS:
        from .config import LOCAL
        return LOCAL / "models" / selected
    path = Path(selected)
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("The selected detector is missing. Restore its weights or include a shared detector.")
    return path


def detector_name(selected):
    if selected == "bundled":
        return "WatchMyCat shared detector"
    if selected in STARTER_MODELS:
        return "Starter detector · first training"
    return "Current trained cat detector"
