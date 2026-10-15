"""Desktop entry point for Cat Care Monitor."""
import argparse
import json
import time
from catcare.config import LOCAL, load_settings, profile_stem
from catcare.storage import Store


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        if "rtsp://" in message.lower() or "rtsps://" in message.lower():
            message = "Enter camera URLs in the desktop dialog, not on the command line."
        super().error(message)


def main():
    parser = SafeArgumentParser(description="Local RTSP monitoring for one cat and three stations.")
    parser.add_argument("--summary", action="store_true", help="Show the current cat's rolling 24-hour summary")
    parser.add_argument("--preview", action="store_true", help="Preview detections without recording visits")
    parser.add_argument("--full-frame", action="store_true", help="Analyze the entire camera frame")
    args = parser.parse_args()
    if args.summary:
        config = load_settings()
        stem = profile_stem(config["cat_label"])
        path = LOCAL / "data" / f"visits_{stem}.sqlite3"
        if not path.exists():
            print("No recorded visits yet for this cat profile.")
            return
        store = Store(path, recover=False)
        try:
            print(json.dumps(store.summary(time.time()), indent=2))
            store.export(LOCAL / "reports" / f"visits_{stem}.csv")
        finally:
            store.close()
        return
    try:
        from catcare.ui import connect_dialog
        result = connect_dialog()
        if result is None:
            return
        config, source = result
        if args.full_frame:
            config["crop"] = False
        from catcare.monitor import run_monitor
        run_monitor(config, source, preview=args.preview)
    except KeyboardInterrupt:
        print("Monitoring stopped.")
    except (ValueError, ImportError) as error:
        print("Could not start:", str(error))
        raise SystemExit(1) from None
    except Exception:
        print("Monitoring stopped after an error. Check the camera, model, and dependencies.")
        print("Camera credentials are omitted from error output.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
