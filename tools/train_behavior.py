"""Train the local temporal classifier on manually reviewed station clips."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catcare.behavior import ACTIVITIES, CLIP_FRAMES
from catcare.behavior_data import ClipLibrary, write_json
from catcare.behavior_model import FrameEncoder, fit_temporal_head
from catcare.config import LOCAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    def progress(phase, **values):
        write_json(args.progress_file, {"phase": phase, "epochs": args.epochs, **values})

    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        library = ClipLibrary(dataset["source"])
        progress("preparing", message="Loading official visual features. The first run downloads a 10 MB encoder.")
        encoder = FrameEncoder(LOCAL / "models", args.device)
        features, targets, splits = [], [], {}
        total = sum(len(row["clips"]) for row in dataset["splits"].values())
        for split in ("train", "val", "test"):
            splits[split] = []
            for row in dataset["splits"][split]["clips"]:
                frames = [library.frame_path(row["id"], index).read_bytes() for index in range(CLIP_FRAMES)]
                splits[split].append(len(features))
                features.append(encoder.encode(frames))
                targets.append(ACTIVITIES.index(row["label"]))
                progress("preparing", prepared=len(features), total=total,
                         message=f"Preparing reviewed clips: {len(features)} / {total}.")
        fit_temporal_head(features, targets, splits, args.output, args.epochs,
                          str(encoder.device), progress)
    except Exception as error:
        from catcare.diagnostics import redact
        progress("failed", message=redact(f"{type(error).__name__}: {error}"))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
