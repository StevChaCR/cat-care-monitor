"""Review frames and manually draw cat boxes; no automatic ground-truth labels."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catcare.config import LOCAL
from catcare.display import fit_frame, screen_bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=LOCAL / "captures")
    parser.add_argument("--review-existing", action="store_true")
    args = parser.parse_args()
    import cv2
    images = sorted(p for p in args.source.glob("*/images/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit("No frames found. Press S in the monitor to collect review frames.")
    window = "Label cats | B add box | N empty scene | S save and next | Q quit"
    bounds = screen_bounds()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window, 40, 40)
    for path in images:
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        if label.exists() and not args.review_existing:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            print("Unreadable image:", path.name)
            continue
        # Labels are normalized, so selections on this fitted FULL image remain
        # valid for the original-resolution training file.
        display = fit_frame(frame, cv2, bounds)
        height, width = display.shape[:2]
        boxes = []
        reviewed_negative = False
        while True:
            view = display.copy()
            for x, y, w, h in boxes:
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 230, 0), 2)
            cv2.imshow(window, view)
            key = cv2.waitKey(0) & 0xff
            if key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return
            if key in (ord("b"), ord("B")):
                x, y, w, h = cv2.selectROI("Draw the visible cat: ENTER confirms", display, False, False)
                cv2.destroyWindow("Draw the visible cat: ENTER confirms")
                if w and h:
                    boxes.append((x, y, w, h))
                    reviewed_negative = False
            if key in (ord("n"), ord("N")):
                boxes = []
                reviewed_negative = True
            if key in (ord("s"), ord("S")):
                if not boxes and not reviewed_negative:
                    print("Add all visible cat boxes, or press N to confirm a cat-free image.")
                    continue
                label.parent.mkdir(parents=True, exist_ok=True)
                text = "".join(f"0 {(x + w/2)/width:.6f} {(y + h/2)/height:.6f} {w/width:.6f} {h/height:.6f}\n"
                               for x, y, w, h in boxes)
                label.write_text(text, encoding="utf-8")
                break
    cv2.destroyAllWindows()
    print("Manual annotation complete.")


if __name__ == "__main__":
    main()
