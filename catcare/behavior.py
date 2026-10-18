"""Conservative activity sessions: missing evidence never means a new meal."""
from dataclasses import dataclass, field

ACTIVITIES = ("eating", "drinking", "other")
STATION_ACTIVITY = {"food": "eating", "water": "drinking"}
CLIP_FRAMES = 16
CLIP_FPS = 4


@dataclass
class ActivitySession:
    started_at: float
    last_seen_at: float
    intervals: list = field(default_factory=list)
    status: str = "in_progress"
    uncertain: bool = False
    row_id: int | None = None


class ActivityCounter:
    """Count sustained model evidence; separately measure only observed intervals.

    True: reliable action prediction. False: reliable other action or observed
    departure. None: occlusion, weak prediction, stale video, or multiple cats.
    A detector's held bounding box must never be passed as True.
    """
    def __init__(self, confirm_seconds=2.0, exit_seconds=6.0, maximum_gap=2.5, resume=None):
        self.confirm_seconds, self.exit_seconds = confirm_seconds, exit_seconds
        self.maximum_gap = maximum_gap
        self.session = resume
        self.candidate = None
        self.candidate_intervals = []
        self.last_positive = None
        self.last_timestamp = None
        self.negative_since = None
        self.unknown_since = None
        self.departure_observed = False

    @staticmethod
    def extend(intervals, start, end):
        if end <= start:
            return
        if intervals and abs(intervals[-1][1] - start) < 1e-6:
            intervals[-1][1] = end
        else:
            intervals.append([start, end])

    def observe(self, evidence, timestamp):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return []
        self.last_timestamp = timestamp
        events = []
        if evidence is None:
            self.last_positive = None
            self.candidate = None
            self.candidate_intervals = []
            self.negative_since = None
            if self.unknown_since is None:
                self.unknown_since = timestamp
            if self.session and not self.session.uncertain:
                self.session.uncertain = True
                events.append(("update", self.session))
            return events
        self.unknown_since = None
        if evidence is False:
            self.last_positive = self.candidate = None
            self.candidate_intervals = []
            if self.negative_since is None:
                self.negative_since = timestamp
            if timestamp - self.negative_since >= self.exit_seconds:
                self.departure_observed = True
                if self.session:
                    self.session.status = "completed"
                    events.append(("end", self.session))
                    self.session = None
            return events
        self.negative_since = None
        previous = self.last_positive
        continuous = previous is not None and 0 < timestamp - previous <= self.maximum_gap
        self.last_positive = timestamp
        if self.session:
            if continuous:
                self.extend(self.session.intervals, previous, timestamp)
            else:
                self.session.uncertain = True
            self.session.last_seen_at = timestamp
            events.append(("update", self.session))
            return events
        if self.candidate is None or not continuous:
            self.candidate = timestamp
            self.candidate_intervals = []
        elif continuous:
            self.extend(self.candidate_intervals, previous, timestamp)
        observed = sum(end - start for start, end in self.candidate_intervals)
        if observed >= self.confirm_seconds:
            self.session = ActivitySession(self.candidate, timestamp,
                [list(pair) for pair in self.candidate_intervals], uncertain=not self.departure_observed)
            self.candidate = None
            self.candidate_intervals = []
            events.append(("start", self.session))
        return events

    def interrupt(self):
        self.last_positive = self.candidate = None
        self.candidate_intervals = []
        self.negative_since = None
        if self.session:
            self.session.status = "interrupted"
            self.session.uncertain = True
            return self.session
        return None


def activity_evidence(station, probabilities, threshold=.85, margin=.15):
    """A probability is a model score, not a calibrated safety guarantee."""
    if station not in STATION_ACTIVITY or set(probabilities) != set(ACTIVITIES):
        return None
    import math
    if not all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
               for p in probabilities.values()):
        return None
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    label = ranked[0]
    if probabilities[label] < threshold or probabilities[label] - probabilities[ranked[1]] < margin:
        return None
    if label == "other":
        return False
    return True if label == STATION_ACTIVITY[station] else None
