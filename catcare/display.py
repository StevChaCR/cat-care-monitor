"""Fit camera views to the desktop without shrinking the status text."""
from collections import deque


class FrameRate:
    """Measured new-frame throughput over a short rolling window."""
    def __init__(self):
        self.samples = deque(maxlen=120)

    def record(self, timestamp):
        self.samples.append(timestamp)

    def value(self, now):
        while self.samples and self.samples[0] < now - 2:
            self.samples.popleft()
        if len(self.samples) < 2 or self.samples[-1] <= self.samples[0]:
            return 0.0
        return (len(self.samples) - 1) / (self.samples[-1] - self.samples[0])


def screen_bounds():
    """Reserve space for the taskbar, window borders, and desktop scaling."""
    root = None
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        return (min(1100, max(320, root.winfo_screenwidth() - 80)),
                min(800, max(360, root.winfo_screenheight() - 140)))
    except Exception:
        # Screen discovery is optional, for example in a headless test run.
        return 1100, 740
    finally:
        if root is not None:
            root.destroy()


def fit_dimensions(shape, bounds, upscale=False):
    """Return an aspect-preserving size that includes every image edge."""
    height, width = shape[:2]
    max_width, max_height = bounds
    scale = min(max_width / width, max_height / height)
    if not upscale:
        scale = min(scale, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def fit_frame(frame, cv2, bounds, upscale=False):
    size = fit_dimensions(frame.shape, bounds, upscale)
    if size == (frame.shape[1], frame.shape[0]):
        return frame
    interpolation = cv2.INTER_AREA if size[0] < frame.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(frame, size, interpolation=interpolation)


def normalized_roi(rectangle, display_shape):
    """Map a selection on the fitted image back to the full camera frame."""
    x, y, width, height = rectangle
    image_height, image_width = display_shape[:2]
    return [x / image_width, y / image_height,
            (x + width) / image_width, (y + height) / image_height]


def wrap_status(lines, cv2, max_width, font_scale=0.65):
    """Keep long status messages inside the panel without reducing the font."""
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = f"{current} {word}" if current else word
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
            if current and width > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
    return wrapped


def status_view(frame, lines, cv2, bounds):
    """Resize only video; draw crisp status text at final display resolution."""
    import numpy as np
    width, height = bounds
    margin, line_height = 16, 30
    wrapped = wrap_status(lines, cv2, width - margin * 2)
    panel_height = margin * 2 + line_height * len(wrapped)
    canvas = np.full((height, width, 3), (12, 14, 18), dtype=np.uint8)
    canvas[:panel_height] = (26, 30, 38)
    video = fit_frame(frame, cv2, (width, max(1, height - panel_height)), upscale=True)
    video_height, video_width = video.shape[:2]
    left = (width - video_width) // 2
    top = panel_height + max(0, (height - panel_height - video_height) // 2)
    canvas[top:top + video_height, left:left + video_width] = video
    for row, line in enumerate(wrapped):
        color = (100, 225, 250) if row == 0 else (245, 245, 245)
        cv2.putText(canvas, line, (margin, margin + 20 + row * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas
