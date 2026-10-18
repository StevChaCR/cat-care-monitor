"""Bundle a reviewed cat-only detector without training photos or local paths."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .behavior_data import write_json
from .models import bundled_detector, release_directory

METRICS = ("precision", "recall", "mAP50", "mAP50_95")


def detector_report(weights):
    path = Path(weights).parent.parent / "held_out_metrics.json"
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("Train and evaluate this detector in the laboratory before including it in a release.")
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("The detector evaluation report could not be read.") from None
    if not isinstance(row, dict) or row.get("evaluation_split") != "test":
        raise ValueError("The detector needs held-out test results before bundling.")
    metrics = {key: row.get(key) for key in METRICS}
    if not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
               for value in metrics.values()):
        raise ValueError("The detector evaluation metrics are invalid.")
    expected = row.get("weights_sha256")
    if expected and hashlib.sha256(Path(weights).read_bytes()).hexdigest() != expected:
        raise ValueError("The selected weights changed after evaluation. Evaluate them again before bundling.")
    return row, metrics


def bundle_detector(weights, output=None):
    source = Path(weights).resolve()
    output = Path(output) if output is not None else release_directory()
    if source == (output / "model.pt").resolve():
        bundled_detector(output)
        return json.loads((output / "model.json").read_text(encoding="utf-8"))
    report, metrics = detector_report(source)
    from ultralytics import YOLO
    model = YOLO(str(source))
    if model.task != "detect" or list(model.names.values()) != ["cat"]:
        raise ValueError("Include a trained cat-only detector. Generic multi-class models are not release candidates.")
    size = report.get("image_size", 960)
    if type(size) is not int or not 32 <= size <= 2048:
        raise ValueError("The detector evaluation input size is invalid.")
    # Rebuild the checkpoint metadata instead of publishing its optimizer,
    # dataset paths, trainer arguments, or the developer's filesystem paths.
    public_args = {"task": "detect", "imgsz": size, "single_cls": True}
    model.model.args = dict(public_args)
    if hasattr(model.model, "pt_path"):
        model.model.pt_path = ""
    if hasattr(model.model, "yaml_file"):
        model.model.yaml_file = Path(model.model.yaml_file).name
    if isinstance(getattr(model.model, "yaml", None), dict) and "yaml_file" in model.model.yaml:
        model.model.yaml["yaml_file"] = Path(model.model.yaml["yaml_file"]).name
    model.ckpt = {"train_args": public_args}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="detector-release-", dir=output.parent) as temporary:
        staging = Path(temporary)
        packaged = staging / "model.pt"
        model.save(packaged)
        check = YOLO(str(packaged))
        if check.task != "detect" or list(check.names.values()) != ["cat"]:
            raise ValueError("The packaged detector could not be reloaded.")
        data_counts = {}
        for split in ("train", "val", "test"):
            values = report.get("dataset", {}).get(split, {})
            data_counts[split] = {key: value for key, value in values.items()
                                  if key in ("images", "cat_boxes", "session_count") and type(value) is int and value >= 0}
        metadata = {"schema_version": 1, "file": "model.pt", "task": "detect", "classes": ["cat"],
                    "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                    "sha256": hashlib.sha256(packaged.read_bytes()).hexdigest(), "image_size": size,
                    "license": "AGPL-3.0-only", "evaluation": {"split": "test", "metrics": metrics,
                    "dataset": data_counts, "scope": "held-out recording sessions",
                    "unseen_cats_and_cameras_verified": False}}
        write_json(staging / "model.json", metadata)
        card = ("# WatchMyCat shared cat detector\n\n"
                f"Version: {metadata['version']}\n\n"
                "This detector locates visible cats. It does not identify individual cats or classify eating, drinking, or elimination.\n\n"
                "## Held-out image results\n\n" +
                "\n".join(f"- {key}: {value:.4f}" for key, value in metrics.items()) +
                "\n\nThese results describe the supplied recording-session split. Generalization to unseen cats and cameras has not been certified. "
                "Validate complete independent recordings before claiming broad compatibility.\n\n"
                "The package contains weights and aggregate metrics. Training photos, clips, and camera credentials are not included.\n\n"
                "License: AGPL-3.0-only; retain the project and upstream notices.\n")
        (staging / "MODEL_CARD.md").write_text(card, encoding="utf-8")
        bundled_detector(staging)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "MODEL_CARD.md", "model.json"):
            (staging / name).replace(output / name)
    return metadata
