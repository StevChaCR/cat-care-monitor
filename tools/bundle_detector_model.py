"""Include the currently selected developer detector in a distributable build."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catcare.config import load_settings
from catcare.models import resolve_detector
from catcare.release import bundle_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Evaluated cat-only best.pt; otherwise use the selected detector")
    parser.add_argument("--output", type=Path, help="Default: assets/detector")
    args = parser.parse_args()
    try:
        weights = args.weights if args.weights else resolve_detector(load_settings()["model"])
        row = bundle_detector(weights, args.output)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print("Shared detector version:", row["version"])
    print("Normal mode loads this detector automatically on new installations.")
    print("The existing trained weights and private data remain in their original locations.")


if __name__ == "__main__":
    main()
