"""Useful runtime errors without camera URLs, credentials, or full tracebacks."""
import json
import re
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit

URL = re.compile(r"(?:rtsp|rtsps|https?|ftp)://[^\s<>\"']+", re.I)
GPU_ERRORS = ("cuda", "cudnn", "cublas", "no kernel image", "not compiled with cuda",
              "device-side assert", "sm_61", "sm_60")


def exception_text(error):
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 3:
        seen.add(id(error))
        try:
            result.append(f"{type(error).__name__}: {error}")
        except Exception:
            result.append(type(error).__name__)
        error = error.__cause__ or error.__context__
    return " | ".join(result)


def gpu_failure(error):
    return any(word in exception_text(error).lower() for word in GPU_ERRORS)


def redact(text, source=""):
    # Replace the whole URI before any component replacement. Also cover native
    # errors that print an individual credential instead of the complete URL.
    secrets = [source, unquote(source)] if source else []
    try:
        parsed = urlsplit(source)
        secrets += [parsed.username, parsed.password, parsed.hostname]
        secrets += [unquote(value) for value in secrets if value]
    except (ValueError, TypeError):
        pass
    text = URL.sub("[redacted URL]", str(text))
    for value in sorted({value for value in secrets if value}, key=len, reverse=True):
        text = text.replace(value, "[redacted]")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text[:1200]


def diagnostic(stage, error, source=""):
    detail = redact(exception_text(error), source)
    raw = exception_text(error).lower()
    code = f"{stage}_failed"
    if stage == "camera":
        message = "No camera video was received. Verify the complete RTSP URL in VLC and close other camera viewers."
        if "ffmpeg" in raw and "unavailable" in raw:
            code = "opencv_ffmpeg_unavailable"
            message = "This OpenCV installation has no FFmpeg camera backend. Reinstall the project requirements."
        else:
            code = "camera_no_video"
    elif isinstance(error, (ModuleNotFoundError, ImportError)):
        code = "dependency_unavailable"
        message = "A model dependency could not load. Run start_web_windows.bat again and check the details below."
    elif gpu_failure(error):
        code = "gpu_unavailable"
        message = "GPU processing failed. Camera preview can continue; select CPU or use the automatic CPU fallback."
    elif stage == "model" and any(word in raw for word in ("download", "certificate", "ssl", "connection", "urlerror")):
        code = "model_download_failed"
        message = "The detector weights could not be downloaded. Check internet access, then reconnect the camera."
    elif stage == "model":
        code = "model_load_failed"
        message = "The detector could not load. Camera preview can continue. Check the model and details below."
    elif stage == "detection":
        message = "Detection stopped. Camera preview can continue. Check the details or reconnect with Processor set to CPU."
    elif stage == "storage":
        message = "The visit database could not be opened or updated. Check that the local folder is writable."
    else:
        message = "A local camera task failed. Check the diagnostic details below."
    return {"id": uuid.uuid4().hex[:10], "time": datetime.now(timezone.utc).isoformat(),
            "stage": stage, "code": code, "message": message, "detail": detail}


def save_diagnostic(folder, row):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "latest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass  # The browser still receives the safe error if the disk is full.
    print(f"Cat Care [{row['code']}] {row['detail']}")
