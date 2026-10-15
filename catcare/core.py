"""Visit state machines and station geometry, independent of video libraries."""
from dataclasses import dataclass

STATIONS = ("litter_box", "food", "water")


@dataclass
class Visit:
    entered_at: float
    last_seen_at: float
    status: str = "in_progress"


class VisitCounter:
    def __init__(self, confirm_seconds=2.0, exit_seconds=3.0, entry_gap_seconds=2.0, arm_seconds=8.0):
        self.confirm_seconds = confirm_seconds
        self.exit_seconds = exit_seconds
        self.entry_gap_seconds = entry_gap_seconds
        self.arm_seconds = arm_seconds
        self.armed = False
        self.visit = None
        self.candidate = None
        self.candidate_last_seen = None
        self.absent_since = None
        self.unknown_since = None
        self.requires_departure = False

    def observe(self, present, timestamp):
        events = []
        if present is None:
            if (self.candidate_last_seen is not None and
                    timestamp - self.candidate_last_seen > self.entry_gap_seconds):
                self.candidate = self.candidate_last_seen = None
            self.absent_since = None
            if self.unknown_since is None:
                self.unknown_since = timestamp
            if timestamp - self.unknown_since >= self.entry_gap_seconds:
                if self.visit:
                    # A lost detection is not evidence that the cat departed.
                    # Retain this visit and its last measured time until reacquisition
                    # or an observed exit. Long gaps remain flagged as uncertain.
                    self.visit.status = "uncertain"
            return events
        self.unknown_since = None
        if present:
            self.absent_since = None
            if self.visit:
                self.visit.last_seen_at = timestamp
            elif self.armed:
                if (self.candidate is None or
                        timestamp - self.candidate_last_seen > self.entry_gap_seconds):
                    self.candidate = timestamp
                self.candidate_last_seen = timestamp
                if timestamp - self.candidate >= self.confirm_seconds:
                    self.visit = Visit(self.candidate, timestamp)
                    self.candidate = self.candidate_last_seen = None
                    events.append(("start", self.visit))
        else:
            if (self.candidate_last_seen is not None and
                    timestamp - self.candidate_last_seen > self.entry_gap_seconds):
                self.candidate = self.candidate_last_seen = None
            if self.absent_since is None:
                self.absent_since = timestamp
            threshold = self.exit_seconds if self.visit or self.requires_departure else self.arm_seconds
            if timestamp - self.absent_since >= threshold:
                self.armed = True
                self.requires_departure = False
                if self.visit:
                    if self.visit.status == "in_progress":
                        self.visit.status = "completed"
                    events.append(("end", self.visit))
                    self.visit = None
        return events

    def interrupt(self, status="interrupted"):
        visit = self.visit
        self.requires_departure = self.requires_departure or visit is not None
        if visit:
            visit.status = status
        self.visit = self.candidate = self.candidate_last_seen = None
        self.absent_since = self.unknown_since = None
        self.armed = False
        return visit


def box_is_clear_of_zone(box, rectangle, margin=0.20):
    """Require an observed cat beyond the zone's deadband before closing a visit."""
    left, top, right, bottom = box
    x1, y1, x2, y2 = rectangle
    pad_x = max(4, (x2 - x1) * margin)
    pad_y = max(4, (y2 - y1) * margin)
    return (right < x1 - pad_x or left > x2 + pad_x or
            bottom < y1 - pad_y or top > y2 + pad_y)


def station_presence(counter, station, cats, rectangles, occupied, ambiguous):
    """Missing detections and boundary jitter cannot certify a physical exit."""
    if len(cats) > 1 or station in ambiguous:
        return None
    if not cats:
        if counter.visit or counter.requires_departure or counter.candidate is not None:
            return None
        return False  # Initial empty-zone arming, before any visit is counted.
    if station == occupied:
        return True
    if counter.visit or counter.requires_departure:
        return False if box_is_clear_of_zone(cats[0], rectangles[station]) else None
    return False


def validate_zones(zones):
    """Reject malformed, empty, or overlapping normalized station rectangles."""
    import math
    if not isinstance(zones, dict) or not zones:
        raise ValueError("Select at least one station.")
    for name, rectangle in zones.items():
        if name not in STATIONS or len(rectangle) != 4:
            raise ValueError("Invalid station configuration.")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in rectangle):
            raise ValueError("Zone coordinates must be finite numbers.")
        x1, y1, x2, y2 = rectangle
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Zones must be nonempty and inside the camera frame.")
    rectangles = list(zones.values())
    for index, a in enumerate(rectangles):
        for b in rectangles[index + 1:]:
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                raise ValueError("Station zones must not overlap. Select separate areas.")


def pixel_zones(zones, width, height):
    return {name: tuple(int(v * scale) for v, scale in zip(rect, (width, height, width, height)))
            for name, rect in zones.items()}


def assign_station(box, rectangles, minimum_overlap=0.10, ambiguity_margin=0.05):
    """Assign one cat to one station, or report ambiguous station occupancy.

    Food and water measure presence near a station, not ingestion.
    The litter-box gate also requires a lower-body point inside its zone.
    """
    left, top, right, bottom = box
    area = max(0, right - left) * max(0, bottom - top)
    if area <= 0:
        return None, set()
    scores = []
    for name, (x1, y1, x2, y2) in rectangles.items():
        overlap = max(0, min(right, x2) - max(left, x1)) * max(0, min(bottom, y2) - max(top, y1))
        score = overlap / area
        if name == "litter_box":
            point_x, point_y = (left + right) / 2, top + 0.85 * (bottom - top)
            if not (x1 <= point_x <= x2 and y1 <= point_y <= y2):
                continue
        if score >= minimum_overlap:
            scores.append((score, name))
    scores.sort(reverse=True)
    if not scores:
        return None, set()
    if len(scores) > 1 and scores[0][0] - scores[1][0] < ambiguity_margin:
        return None, {name for score, name in scores if scores[0][0] - score < ambiguity_margin}
    return scores[0][1], set()
