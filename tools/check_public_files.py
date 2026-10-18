"""Check staged/tracked files before publishing; never print secret matches."""
import re
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
URI = re.compile(r"rtsps?://([^\s'\"<>:@]+):([^\s'\"<>@]+)@", re.IGNORECASE)
SAFE_EXAMPLES = {("unit", "fake"), ("username", "password"), ("user", "password"), ("test", "test")}


def valid_release_resource(name):
    try:
        if name == "assets/detector/model.pt":
            from catcare.models import bundled_detector
            bundled_detector(ROOT / "assets" / "detector")
            return True
        if name in ("assets/behavior/weights.npz", "assets/behavior/encoder.pth"):
            from catcare.behavior_model import ENCODER_SHA256
            folder = ROOT / "assets" / "behavior"
            manifest = folder / "model.json"
            if not manifest.is_file() or manifest.stat().st_size > 65536:
                return False
            row = json.loads(manifest.read_text(encoding="utf-8"))
            if row.get("format") != "watchmycat-activity-v1" or row.get("classes") != ["eating", "drinking", "other"]:
                return False
            for filename, maximum, expected in (("weights.npz", 8_000_000, row.get("weights_sha256")),
                    ("encoder.pth", 30_000_000, ENCODER_SHA256)):
                path = folder / filename
                if not path.is_file() or not 0 < path.stat().st_size <= maximum:
                    return False
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    return False
            return row.get("encoder_sha256") == ENCODER_SHA256
    except (ValueError, OSError, AttributeError):
        return False
    return False


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise SystemExit("Run git init and git add . first, then check the files before committing.")
    failures = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if not name:
            continue
        path = ROOT / name
        if name in ("assets/detector/model.pt", "assets/behavior/weights.npz", "assets/behavior/encoder.pth"):
            if not valid_release_resource(name):
                failures.append(name)
            continue
        if name.startswith(("local/", ".venv/")) or path.suffix.lower() in {
            ".pt", ".pth", ".npz", ".weights", ".sqlite3", ".db", ".csv", ".mp4", ".avi", ".mov",
        } or path.name == ".env" or path.name.startswith(".env."):
            failures.append(name)
            continue
        if path.is_file() and path.stat().st_size < 2_000_000:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any((match.group(1), match.group(2)) not in SAFE_EXAMPLES and
                   not ("{" in match.group(1) and "{" in match.group(2))
                   for match in URI.finditer(text)):
                failures.append(name)
    if failures:
        print("Remove private data or embedded RTSP credentials from these tracked files:")
        print("\n".join(sorted(set(failures))))
        raise SystemExit(1)
    print("Public-file check passed. No private runtime files or embedded nonexample RTSP credentials found.")


if __name__ == "__main__":
    main()
