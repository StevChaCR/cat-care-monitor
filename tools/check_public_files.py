"""Check staged/tracked files before publishing; never print secret matches."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URI = re.compile(r"rtsps?://([^\s'\"<>:@]+):([^\s'\"<>@]+)@", re.IGNORECASE)
SAFE_EXAMPLES = {("unit", "fake"), ("username", "password"), ("user", "password"), ("test", "test")}


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise SystemExit("Run git init and git add . first, then check the files before committing.")
    failures = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if not name:
            continue
        path = ROOT / name
        if name.startswith(("local/", ".venv/")) or path.suffix.lower() in {
            ".pt", ".weights", ".sqlite3", ".db", ".csv", ".mp4", ".avi", ".mov",
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
