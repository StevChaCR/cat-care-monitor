"""Prepare manually labeled data with recording-session splits."""
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_labels(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError("Labels must use class 0 and four normalized box coordinates.")
        try:
            x, y, width, height = map(float, fields[1:])
        except ValueError:
            raise ValueError("Label coordinates must be numbers.") from None
        if (not all(math.isfinite(v) for v in (x, y, width, height)) or
                not (0 < width <= 1 and 0 < height <= 1) or
                x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or
                y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6):
            raise ValueError("A label box is outside its image.")
        rows.append((0, x, y, width, height))
    return rows


def prepare_dataset(source, output, seed=42, split_file=None, inherit_manifest=None):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains data. Choose a new --output directory.")
    groups = []
    seen_hashes = {}
    for session in sorted(source.iterdir() if source.exists() else []):
        if not session.is_dir() or not (session / "images").is_dir():
            continue
        images = sorted(p for p in (session / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            continue
        entries = []
        seen_stems = set()
        for image in images:
            if image.stem in seen_stems:
                raise ValueError("Images in one session must have unique filename stems.")
            seen_stems.add(image.stem)
            label = session / "labels" / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError("Every image needs a reviewed label file; mark empty scenes as negative.")
            rows = read_labels(label)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest] != session.name:
                raise ValueError("Identical images appear in different sessions. Remove duplicate copies before splitting.")
            seen_hashes[digest] = session.name
            entries.append((image, label, len(rows)))
        groups.append((session.name, entries))
    if len(groups) < 3:
        raise ValueError("Use at least three distinct recording sessions: training, validation, and test.")
    frozen = None
    split_path = Path(split_file) if split_file is not None else None
    if split_path is not None and split_path.is_file():
        frozen = json.loads(split_path.read_text(encoding="utf-8"))
    elif inherit_manifest is not None and Path(inherit_manifest).is_file():
        previous = json.loads(Path(inherit_manifest).read_text(encoding="utf-8"))
        frozen = {"schema_version": 1, "seed": previous.get("seed", seed), "assignments": {}}
        for split, row in previous.get("splits", {}).items():
            for session in row.get("sessions", []):
                if session in frozen["assignments"]:
                    raise ValueError("The previous dataset repeats a session across splits.")
                frozen["assignments"][session] = split
    if frozen is not None:
        roles = frozen.get("assignments") if isinstance(frozen, dict) else None
        if (not isinstance(frozen, dict) or frozen.get("schema_version") != 1 or not isinstance(roles, dict) or
                not all(isinstance(name, str) and role in ("train", "val", "test") for name, role in roles.items())):
            raise ValueError("The saved detector split is invalid. Restore it before training.")
        assignments = {split: [] for split in ("train", "val", "test")}
        for name, entries in groups:
            role = roles.setdefault(name, "train")
            assignments[role].append((name, entries))
    else:
        random.Random(seed).shuffle(groups)
        held_out = max(1, round(len(groups) * 0.15))
        assignments = {"test": groups[:held_out], "val": groups[held_out:2 * held_out],
                       "train": groups[2 * held_out:]}
        frozen = {"schema_version": 1, "seed": seed, "assignments": {
            name: split for split, sessions in assignments.items() for name, _ in sessions}}
    for split, sessions in assignments.items():
        if not sessions or sum(count for _, entries in sessions for _, _, count in entries) == 0:
            raise ValueError(f"The {split} split needs at least one labeled cat. Add cat-positive sessions or change --seed.")
    metadata = {"seed": frozen.get("seed", seed), "classes": ["cat"], "splits": {}}
    for split, sessions in assignments.items():
        images_dir, labels_dir = output / "images" / split, output / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_count = box_count = 0
        for group_index, (_, entries) in enumerate(sessions):
            for image, label, count in entries:
                stem = f"session_{group_index:03d}_{image.stem}"
                shutil.copy2(image, images_dir / (stem + image.suffix.lower()))
                shutil.copy2(label, labels_dir / (stem + ".txt"))
                image_count += 1
                box_count += count
        metadata["splits"][split] = {"sessions": [name for name, _ in sessions],
                                      "images": image_count, "cat_boxes": box_count}
    data_yaml = ("path: " + json.dumps(str(output.resolve())) + "\n"
                 "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cat\n")
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = split_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        temporary.replace(split_path)
    return metadata
