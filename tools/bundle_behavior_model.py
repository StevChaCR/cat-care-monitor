"""Include a developer-reviewed activity candidate in a distributable build."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catcare.behavior_data import write_json
from catcare.behavior_model import load_activity_head
from catcare.config import LOCAL, ROOT


def bundle(run, output, encoder=None):
    run, output = Path(run), Path(output)
    manifest_path = run / "model.json"
    _, metadata = load_activity_head(manifest_path)
    if metadata.get("metrics", {}).get("evaluation_split") != "test":
        raise ValueError("Train and evaluate the candidate on held-out test sessions before bundling it.")
    if encoder is None:
        candidates = list((LOCAL / "models" / "torch" / "checkpoints").glob("mobilenet_v3_small-*.pth"))
        if len(candidates) != 1:
            raise ValueError("Run behavior training first, or provide the official encoder checkpoint with --encoder.")
        encoder = candidates[0]
    encoder = Path(encoder)
    if not encoder.is_file() or not 0 < encoder.stat().st_size <= 30_000_000:
        raise ValueError("The official visual encoder checkpoint is missing or exceeds the size limit.")
    # Validate the compatible official architecture with the safe tensor loader.
    from catcare.behavior_model import FrameEncoder
    FrameEncoder(LOCAL / "models", weights_file=encoder)
    output.mkdir(parents=True, exist_ok=True)
    if run.resolve() == output.resolve():
        raise ValueError("Choose an output directory separate from the training run.")
    shutil.copy2(run / "weights.npz", output / "weights.npz")
    shutil.copy2(encoder, output / "encoder.pth")
    metadata["encoder_sha256"] = hashlib.sha256((output / "encoder.pth").read_bytes()).hexdigest()
    write_json(output / "model.json", metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed developer activity training run")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "behavior")
    parser.add_argument("--encoder", type=Path)
    args = parser.parse_args()
    try:
        output = bundle(args.run, args.output, args.encoder)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print("Included activity model:", output)
    print("Normal mode loads it automatically. No end-user labeling or training is required.")
    print("Clip metrics do not replace event-level checks on unseen cats, cameras, and complete recordings.")


if __name__ == "__main__":
    main()
