"""Serve the Cat Care browser UI and authenticated APIs on loopback only."""
import argparse
import io
import json
import mimetypes
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

# Set native decoder privacy options before photo import can load OpenCV.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from catcare import __version__
from catcare.config import ROOT
from catcare.web_runtime import WebRuntime


def make_server(runtime, port=8765, training_mode=False):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log camera URLs, request bodies, or browser query strings.

        def send(self, code, body=b"", content_type="application/json"):
            self.send_response(code)
            if code != 204:
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def reply(self, value, code=200):
            self.send(code, json.dumps(value).encode())

        def guarded(self, api=False):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                self.reply({"error": "Use the local application address."}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in hosts}:
                self.reply({"error": "Open the local app to access your camera."}, 403)
                return False
            if api and not secrets.compare_digest(self.headers.get("X-CatCare-Token", ""), token):
                self.reply({"error": "Reload the local app to reconnect."}, 403)
                return False
            return True

        def body(self, maximum=1_000_000):
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid request size.") from None
            if size <= 0 or size > maximum:
                raise ValueError("The file or request is too large.")
            return self.rfile.read(size)

        def do_OPTIONS(self):
            self.reply({"error": "Cross-origin camera access is disabled. Open the local app."}, 403)

        def do_GET(self):
            path = urlsplit(self.path).path
            if not self.guarded(path.startswith("/api/")):
                return
            if not training_mode and (path == "/api/library" or path.startswith(("/api/image/", "/api/thumbnail/", "/api/behavior/frame/")) or path in ("/api/behavior/library", "/api/behavior/readiness")):
                self.reply({"error": "The training laboratory is disabled in normal mode."}, 403)
                return
            try:
                if path == "/api/status":
                    self.reply({"version": __version__, **runtime.status()})
                elif path == "/api/frame.jpg":
                    frame = runtime.jpeg()
                    if frame is None:
                        # No frame during startup/reconnection is an expected
                        # state. Specific failures are reported by /api/status.
                        self.send(204)
                    else:
                        self.send(200, frame, "image/jpeg")
                elif path == "/api/library":
                    self.reply({"sessions": runtime.library.sessions(), "images": runtime.library.images(),
                                "trash": runtime.library.trashed_images()})
                elif path == "/api/behavior/library":
                    from catcare.behavior_data import behavior_readiness
                    library = runtime.behavior_library
                    sessions, clips = library.sessions(), library.clips()
                    self.reply({"sessions": sessions, "clips": clips, "trash": library.trashed_clips(),
                                "readiness": behavior_readiness(library.root, rows=clips, sessions=sessions)})
                elif path == "/api/behavior/readiness":
                    from catcare.behavior_data import behavior_readiness
                    self.reply(behavior_readiness(runtime.behavior_library.root))
                elif path.startswith("/api/behavior/frame/"):
                    key, index = path.removeprefix("/api/behavior/frame/").rsplit("/", 1)
                    trashed = parse_qs(urlsplit(self.path).query).get("trash") == ["1"]
                    self.send(200, runtime.behavior_library.frame_path(key, int(index), trashed=trashed).read_bytes(), "image/jpeg")
                elif path == "/api/behavior/report.csv":
                    from catcare.config import LOCAL, profile_stem
                    report = LOCAL / "reports" / f"activities_{profile_stem(runtime.config['cat_label'])}.csv"
                    if report.is_file():
                        self.send(200, report.read_bytes(), "text/csv; charset=utf-8")
                    else:
                        self.reply({"error": "No activity report has been recorded yet."}, 404)
                elif path.startswith("/api/image/"):
                    trashed = parse_qs(urlsplit(self.path).query).get("trash") == ["1"]
                    self.send(200, runtime.library.image_path(path.removeprefix("/api/image/"), trashed=trashed).read_bytes(), "image/jpeg")
                elif path.startswith("/api/thumbnail/"):
                    from PIL import Image
                    trashed = parse_qs(urlsplit(self.path).query).get("trash") == ["1"]
                    with Image.open(runtime.library.image_path(path.removeprefix("/api/thumbnail/"), trashed=trashed)) as original:
                        original.thumbnail((360, 240))
                        output = io.BytesIO()
                        original.save(output, format="JPEG", quality=75)
                    self.send(200, output.getvalue(), "image/jpeg")
                elif path == "/api/report.csv":
                    from catcare.config import LOCAL, profile_stem
                    report = LOCAL / "reports" / f"visits_{profile_stem(runtime.config['cat_label'])}.csv"
                    if report.is_file():
                        self.send(200, report.read_bytes(), "text/csv; charset=utf-8")
                    else:
                        self.reply({"error": "No visit report has been recorded yet."}, 404)
                else:
                    files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
                             "/styles.css": "styles.css", "/behavior.css": "behavior.css", "/icon.svg": "icon.svg"}
                    filename = files.get(path)
                    if filename is None:
                        self.reply({"error": "Not found."}, 404)
                        return
                    data = (ROOT / "docs" / filename).read_bytes()
                    if filename == "index.html":
                        data = data.replace(b"__ENGINE_TOKEN__", token.encode())
                        data = data.replace(b"__TRAINING_MODE__", b"true" if training_mode else b"false")
                    self.send(200, data, mimetypes.guess_type(filename)[0] or "application/octet-stream")
            except ValueError as error:
                self.reply({"error": str(error)}, 400)
            except Exception:
                self.reply({"error": "Could not read local application data."}, 500)

        def do_POST(self):
            if not self.guarded(api=True):
                return
            path = urlsplit(self.path).path
            lab_routes = {"/api/collect", "/api/sessions", "/api/labels", "/api/train", "/api/train/cancel", "/api/train/activate", "/api/train/bundle", "/api/performance"}
            if not training_mode and (path in lab_routes or path.startswith(("/api/upload/", "/api/images/", "/api/behavior/"))):
                self.reply({"error": "Open the developer training launcher to use the laboratory."}, 403)
                return
            try:
                if path.startswith("/api/upload/"):
                    session_id = path.removeprefix("/api/upload/")
                    body = self.body(12_000_000)
                    from PIL import Image
                    with Image.open(io.BytesIO(body)) as image:
                        if image.format not in ("JPEG", "PNG") or image.width * image.height > 16_000_000:
                            raise ValueError("Use a JPEG or PNG up to 16 megapixels and 12 MB.")
                    import cv2
                    import numpy as np
                    frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("That file could not be opened as an image.")
                    ok, jpeg = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise ValueError("That image could not be imported.")
                    self.reply(runtime.library.add_image(session_id, jpeg.tobytes(), frame.shape[1], frame.shape[0]))
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send JSON from the local application.")
                data = json.loads(self.body())
                if not isinstance(data, dict):
                    raise ValueError("Invalid request.")
                if path == "/api/settings":
                    runtime.settings(data)
                elif path == "/api/connect":
                    runtime.connect(data.get("url", ""))
                elif path == "/api/disconnect":
                    runtime.disconnect()
                elif path == "/api/zones":
                    runtime.zones(data.get("zones"))
                elif path == "/api/collect":
                    runtime.collect(data.get("enabled") is True)
                elif path == "/api/sessions":
                    self.reply(runtime.library.create_session(data.get("name", "")))
                    return
                elif path == "/api/labels":
                    runtime.label_image(data.get("id"), data.get("boxes"), data.get("negative", False))
                elif path in ("/api/images/trash", "/api/images/restore", "/api/images/purge"):
                    count = runtime.manage_images(path.rsplit("/", 1)[1], data.get("ids"))
                    self.reply({"ok": True, "count": count})
                    return
                elif path == "/api/train":
                    runtime.start_training(data.get("epochs", 60))
                elif path == "/api/train/cancel":
                    runtime.training.cancel()
                elif path == "/api/train/activate":
                    runtime.activate_model()
                elif path == "/api/train/bundle":
                    row = runtime.bundle_model()
                    self.reply({"version": row["version"], "sha256": row["sha256"]})
                    return
                elif path == "/api/behavior/sessions":
                    self.reply(runtime.behavior_library.create_session(data.get("name", "")))
                    return
                elif path == "/api/behavior/collect":
                    runtime.collect_behavior(data.get("enabled") is True, data.get("session"), data.get("station", "both"))
                elif path == "/api/performance":
                    runtime.set_inference_size(data.get("image_size"))
                elif path == "/api/behavior/labels":
                    runtime.label_behavior_clip(data.get("id"), data.get("label"))
                elif path in ("/api/behavior/clips/trash", "/api/behavior/clips/restore", "/api/behavior/clips/purge"):
                    count = runtime.manage_behavior_clips(path.rsplit("/", 1)[1], data.get("ids"))
                    self.reply({"ok": True, "count": count})
                    return
                elif path == "/api/behavior/train":
                    runtime.start_behavior_training(data.get("epochs", 60))
                elif path == "/api/behavior/train/cancel":
                    runtime.behavior_training.cancel()
                elif path == "/api/behavior/train/activate":
                    runtime.activate_behavior_model()
                elif path == "/api/behavior/model/bundled":
                    runtime.activate_behavior_model(bundled=True)
                else:
                    self.reply({"error": "Not found."}, 404)
                    return
                self.reply({"ok": True})
            except (ValueError, TypeError, KeyError) as error:
                self.reply({"error": str(error)}, 400)
            except Exception:
                self.reply({"error": "Could not complete that action. Check the selected files and local dependencies."}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="Open Cat Care Monitor in your browser, using a private local engine.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the local interface automatically")
    parser.add_argument("--training", action="store_true", help="Enable the developer-only image and activity training laboratory")
    args = parser.parse_args()
    runtime = WebRuntime()
    try:
        server = make_server(runtime, args.port, training_mode=args.training)
    except OSError:
        raise SystemExit("The local port is busy. Close the other Cat Care window or choose --port 8766.") from None
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Cat Care Monitor {__version__}: {url}")
    print("Keep this terminal open. Press Ctrl+C to stop. Camera credentials are never saved.")
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
