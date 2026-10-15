import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catcare.config import LOCAL
from catcare.dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Split manually labeled data by recording session.")
    parser.add_argument("--source", type=Path, default=LOCAL / "captures")
    parser.add_argument("--output", type=Path, default=LOCAL / "dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_dataset(args.source, args.output, args.seed)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
