"""ByteTrack settings and a bounded display hold, independent of video libraries."""
from dataclasses import dataclass
import json


def tracker_settings(confidence):
    low = min(.05, confidence / 2)
    return {"tracker_type": "bytetrack", "track_high_thresh": confidence,
            "track_low_thresh": low, "new_track_thresh": confidence,
            "track_buffer": 120, "match_thresh": .8, "fuse_score": True}


def write_tracker_settings(path, confidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON is a YAML subset and avoids a YAML dependency in the core test suite.
    path.write_text(json.dumps(tracker_settings(confidence), indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class TrackView:
    track_id: int
    box: tuple
    confidence: float
    last_seen_at: float
    estimated: bool = False


class CatTrackCache:
    """Show the last box briefly; held boxes are never presence observations."""
    def __init__(self, hold_seconds=5.0):
        self.hold_seconds = hold_seconds
        self.recent = {}

    def update(self, observations, timestamp):
        if observations:
            # Do not display a ghost ID beside a newly matched visible cat.
            self.recent = {track_id: TrackView(track_id, tuple(box), score, timestamp)
                           for track_id, box, score in observations}
            return tuple(self.recent.values())
        self.recent = {key: value for key, value in self.recent.items()
                       if timestamp - value.last_seen_at <= self.hold_seconds}
        return tuple(TrackView(value.track_id, value.box, value.confidence,
                               value.last_seen_at, True) for value in self.recent.values())
