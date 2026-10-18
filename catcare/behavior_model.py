"""Frozen official image features plus a locally trained temporal activity head.

User activity weights contain numeric NumPy arrays, never pickled Python code.
The ImageNet encoder does not recognize cat activities until this head is trained.
"""
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from .behavior import ACTIVITIES, CLIP_FRAMES
from .behavior_data import write_json

ENCODER = "mobilenet_v3_small_imagenet1k_v1"
ENCODER_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
FEATURES = 576


class FrameEncoder:
    def __init__(self, cache, device="cpu", pretrained=True, weights_file=None):
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        torch.set_num_threads(min(4, os.cpu_count() or 2))
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda:0" if device == "0" else device)
        torch.hub.set_dir(str(Path(cache) / "torch"))
        official_cache = Path(cache) / "torch" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
        checked = Path(weights_file) if weights_file is not None else official_cache
        if checked.is_file() and hashlib.sha256(checked.read_bytes()).hexdigest() != ENCODER_SHA256:
            raise ValueError("The official visual encoder failed its SHA-256 integrity check.")
        network = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained and weights_file is None else None,
                                    progress=False)
        if pretrained and checked.is_file() and hashlib.sha256(checked.read_bytes()).hexdigest() != ENCODER_SHA256:
            raise ValueError("The downloaded visual encoder failed its SHA-256 integrity check.")
        if weights_file is not None:
            state = torch.load(weights_file, map_location="cpu", weights_only=True)
            network.load_state_dict(state, strict=True)
        self.network = torch.nn.Sequential(network.features, network.avgpool, torch.nn.Flatten()).to(self.device).eval()
        self.network.requires_grad_(False)

    def encode(self, frames):
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        arrays = []
        for frame in frames:
            with Image.open(io.BytesIO(frame)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("Behavior frame exceeds the image size limit.")
                # Preserve both the cat and bowl; a center crop can remove them.
                image = ImageOps.pad(image.convert("RGB"), (224, 224), method=Image.Resampling.BILINEAR)
                arrays.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255)
        batch = torch.from_numpy(np.stack(arrays)).to(self.device)
        mean = batch.new_tensor([.485, .456, .406])[None, :, None, None]
        std = batch.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            parts = [self.network((batch[index:index + 8] - mean) / std).cpu()
                     for index in range(0, len(batch), 8)]
        return torch.cat(parts).numpy()


def temporal_head():
    import torch
    return torch.nn.Sequential(
        torch.nn.Conv1d(FEATURES, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Dropout(.3),
        torch.nn.Conv1d(64, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(64, len(ACTIVITIES)))


def save_activity_model(folder, head, metrics):
    import numpy as np
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "weights.npz"
    temporary = folder / "weights.tmp"
    with temporary.open("wb") as output:
        np.savez_compressed(output, **{name: value.detach().cpu().numpy() for name, value in head.state_dict().items()})
    temporary.replace(weights)
    manifest = {"format": "watchmycat-activity-v1", "encoder": ENCODER,
                "frames": CLIP_FRAMES, "features": FEATURES, "classes": list(ACTIVITIES),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "metrics": metrics}
    write_json(folder / "model.json", manifest)
    return folder / "model.json"


def load_activity_head(manifest_path, device="cpu"):
    import numpy as np
    import torch
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "watchmycat-activity-v1" or manifest.get("encoder") != ENCODER or
            manifest.get("classes") != list(ACTIVITIES) or manifest.get("frames") != CLIP_FRAMES or
            manifest.get("features") != FEATURES):
        raise ValueError("Unsupported activity model. Train one using this application.")
    weights = path.parent / "weights.npz"
    if weights.stat().st_size > 8_000_000 or hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("The activity model is incomplete or its checksum changed.")
    with zipfile.ZipFile(weights) as archive:
        if sum(item.file_size for item in archive.infolist()) > 8_000_000:
            raise ValueError("Activity model arrays exceed the size limit.")
    head = temporal_head()
    expected = head.state_dict()
    with np.load(weights, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError("Invalid activity model parameters.")
        values = {}
        for name, value in expected.items():
            array = arrays[name]
            if array.shape != tuple(value.shape) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Invalid activity model weights.")
            values[name] = torch.from_numpy(array.copy())
    head.load_state_dict(values, strict=True)
    return head.to(device).eval(), manifest


class ActivityClassifier:
    def __init__(self, manifest, cache, device):
        path = Path(manifest)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        encoder_file = None
        if metadata.get("encoder_sha256"):
            encoder_file = path.parent / "encoder.pth"
            if encoder_file.stat().st_size > 30_000_000 or hashlib.sha256(encoder_file.read_bytes()).hexdigest() != metadata["encoder_sha256"]:
                raise ValueError("The included visual encoder failed its integrity check.")
        self.encoder = FrameEncoder(cache, device, weights_file=encoder_file)
        self.head, self.manifest = load_activity_head(manifest, self.encoder.device)

    def predict(self, frames):
        import torch
        features = self.encoder.encode(frames)
        batch = torch.from_numpy(features).transpose(0, 1).unsqueeze(0).to(self.encoder.device)
        with torch.inference_mode():
            values = self.head(batch).softmax(dim=1)[0].cpu().tolist()
        return dict(zip(ACTIVITIES, values))


def classification_metrics(targets, predictions):
    matrix = [[0] * len(ACTIVITIES) for _ in ACTIVITIES]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    rows = {}
    for index, label in enumerate(ACTIVITIES):
        true_positive = matrix[index][index]
        precision = true_positive / max(1, sum(row[index] for row in matrix))
        recall = true_positive / max(1, sum(matrix[index]))
        rows[label] = {"precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / max(1e-12, precision + recall),
                       "examples": sum(matrix[index])}
    return {"accuracy": sum(matrix[i][i] for i in range(len(ACTIVITIES))) / max(1, len(targets)),
            "macro_f1": sum(row["f1"] for row in rows.values()) / len(ACTIVITIES),
            "per_class": rows, "confusion_matrix": matrix, "clips": len(targets)}


def fit_temporal_head(features, targets, splits, folder, epochs, device, progress):
    """Frozen features make this small personal-data training practical locally."""
    import copy
    import numpy as np
    import torch
    torch.manual_seed(42)
    device = torch.device("cuda:0" if device == "0" else device)
    head = temporal_head().to(device)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).transpose(1, 2).to(device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    train = torch.tensor(splits["train"], dtype=torch.long, device=device)
    counts = torch.bincount(y[train], minlength=len(ACTIVITIES)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts.clamp(min=1)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    best, best_f1, best_loss, stale = None, -1., float("inf"), 0

    def evaluate(indices):
        head.eval()
        with torch.inference_mode():
            logits = head(x[indices])
            loss = float(loss_fn(logits, y[indices]).cpu())
            predicted = logits.argmax(dim=1).cpu().tolist()
        return classification_metrics(y[indices].cpu().tolist(), predicted), loss

    final_epoch = 0
    for epoch in range(epochs):
        head.train()
        for indices in train[torch.randperm(len(train), device=device)].split(16):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        metrics, validation_loss = evaluate(splits["val"])
        final_epoch = epoch + 1
        progress("training", epoch=final_epoch, validation_f1=metrics["macro_f1"])
        if metrics["macro_f1"] > best_f1 or (metrics["macro_f1"] == best_f1 and validation_loss < best_loss):
            best = copy.deepcopy(head.state_dict())
            best_f1, best_loss, stale = metrics["macro_f1"], validation_loss, 0
        else:
            stale += 1
        if stale >= 10:
            break
    head.load_state_dict(best)
    progress("evaluating", epoch=final_epoch)
    metrics, _ = evaluate(splits["test"])
    metrics.update(evaluation_split="test", validation_macro_f1=best_f1)
    manifest = save_activity_model(folder, head, metrics)
    progress("completed", epoch=final_epoch, weights=str(manifest.resolve()), **metrics)
    return manifest, metrics
