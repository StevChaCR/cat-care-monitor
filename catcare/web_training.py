"""Run local training in a cancellable child process, with real progress."""
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from .config import ROOT
from .dataset import prepare_dataset
from .models import STARTER_MODELS, resolve_detector


class TrainingJob:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.folder = None
        self.phase = "idle"
        self.weights = None

    def status(self):
        with self.lock:
            if self.folder is None:
                return {"phase": "idle"}
            path = self.folder / "progress.json"
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                row = {"phase": "preparing", "epoch": 0}
            if self.phase == "cancelled":
                return {"phase": "cancelled"}
            if self.process is not None and self.process.poll() is not None:
                if self.process.returncode != 0 or row.get("phase") != "completed":
                    self.phase = "failed"
                    return {"phase": "failed", "message": "Training stopped. Check your data and available memory."}
                self.phase = "completed"
                candidate = Path(row.get("weights", ""))
                if candidate.is_file() and candidate.resolve().is_relative_to(self.folder.resolve()):
                    self.weights = candidate
            # Local paths are not part of the browser response.
            return {key: value for key, value in row.items() if key != "weights"}

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def start(self, profile, images, model, epochs, image_size, device):
        with self.lock:
            if self.busy():
                raise ValueError("Training is already running.")
            if not isinstance(epochs, int) or not 5 <= epochs <= 200:
                raise ValueError("Choose between 5 and 200 training epochs.")
            folder = Path(profile) / "training" / uuid.uuid4().hex
            selected = resolve_detector(model).resolve()
            inherited = None
            for parent in selected.parents:
                if parent.parent == (Path(profile) / "training").resolve():
                    candidate = parent / "dataset" / "split_manifest.json"
                    if candidate.is_file():
                        inherited = candidate
                    break
            # Validation happens before a child is launched. Incomplete labels and
            # neighboring frames split across fake sessions are not training data.
            prepare_dataset(images, folder / "dataset", split_file=Path(profile) / "detector_split.json",
                            inherit_manifest=inherited)
            self.folder, self.weights, self.phase = folder, None, "preparing"
            progress = folder / "progress.json"
            progress.write_text(json.dumps({"phase": "preparing", "epoch": 0, "epochs": epochs}), encoding="utf-8")
            command = [sys.executable, str(ROOT / "tools" / "train_detector.py"),
                       "--data", str(folder / "dataset" / "data.yaml"), "--model", model if model in STARTER_MODELS else str(selected),
                       "--epochs", str(epochs), "--imgsz", str(image_size), "--batch", "2",
                       "--device", device, "--project", str(folder / "runs"),
                       "--progress-file", str(progress)]
            # No RTSP URI, camera password, or user images are put on this command line.
            with (folder / "training.log").open("w", encoding="utf-8") as log:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT)

    def cancel(self):
        with self.lock:
            if self.busy():
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                self.phase = "cancelled"


class BehaviorTrainingJob(TrainingJob):
    def start(self, profile, images, epochs, device):
        from .behavior_data import behavior_readiness, prepare_behavior_dataset
        with self.lock:
            if self.busy():
                raise ValueError("Behavior training is already running.")
            if type(epochs) is not int or not 5 <= epochs <= 200:
                raise ValueError("Choose between 5 and 200 training epochs.")
            checklist = behavior_readiness(images)
            if not checklist["ready"]:
                raise ValueError(checklist["issues"][0] + " See the Behavior lab dataset checklist for all blockers.")
            folder = Path(profile) / "behavior" / "training" / uuid.uuid4().hex
            prepare_behavior_dataset(images, folder / "dataset.json")
            self.folder, self.weights, self.phase = folder, None, "preparing"
            progress = folder / "progress.json"
            progress.write_text(json.dumps({"phase": "preparing", "epoch": 0, "epochs": epochs}), encoding="utf-8")
            command = [sys.executable, str(ROOT / "tools" / "train_behavior.py"),
                       "--dataset", str(folder / "dataset.json"), "--output", str(folder),
                       "--epochs", str(epochs), "--device", device, "--progress-file", str(progress)]
            with (folder / "training.log").open("w", encoding="utf-8") as log:
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def status(self):
        row = super().status()
        if row.get("phase") == "failed" and self.folder:
            try:
                details = json.loads((self.folder / "progress.json").read_text(encoding="utf-8"))
                if details.get("phase") == "failed":
                    row["message"] = details.get("message", row.get("message"))
            except (OSError, ValueError):
                pass
        return row
