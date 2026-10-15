"""A latest-frame reader: slow inference cannot accumulate an old-video queue."""
import threading
import time


class Camera:
    def __init__(self, source, cv2):
        self._source = source
        self._cv2 = cv2
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._sequence = 0
        self.status = "Connecting"
        self._failure_exception = None
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        cv2 = self._cv2
        registry = getattr(cv2, "videoio_registry", None)
        if registry is not None and not registry.hasBackend(cv2.CAP_FFMPEG):
            with self._lock:
                self._failure_exception = RuntimeError("OpenCV FFmpeg video backend is unavailable.")
            self.status = "Unavailable"
            return
        while not self._stop.is_set():
            capture = None
            try:
                self.status = "Connecting"
                capture = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                ])
                if not capture.isOpened():
                    with self._lock:
                        self._failure_exception = RuntimeError("OpenCV could not open the camera stream.")
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    with self._lock:
                        self._sequence += 1
                        self._latest = (self._sequence, time.time(), frame)
                        self._failure_exception = None
                    self.status = "Live"
            except Exception as error:
                # Native camera exceptions can contain stream credentials.
                with self._lock:
                    self._failure_exception = error.with_traceback(None)
            finally:
                if capture is not None:
                    capture.release()
                with self._lock:
                    self._latest = None
                self.status = "Reconnecting"
            self._stop.wait(2)
        self.status = "Stopped"

    def latest(self, after_sequence=None):
        with self._lock:
            if self._latest is None:
                return None
            sequence, timestamp, frame = self._latest
            if sequence == after_sequence:
                return None
            return sequence, timestamp, frame.copy()

    @property
    def failure_exception(self):
        with self._lock:
            return self._failure_exception

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=7)
        self._source = ""
