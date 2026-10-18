"""SQLite visit records, rolling 24-hour summaries, and observation coverage."""
import csv
import sqlite3
from datetime import datetime, timezone


def iso_timestamp(value):
    return "" if value is None else datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path, recover=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS visits ("
                        "id INTEGER PRIMARY KEY, station TEXT, cat_label TEXT, "
                        "entered_at REAL, last_seen_at REAL, status TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS coverage ("
                        "id INTEGER PRIMARY KEY, started_at REAL, ended_at REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS activities ("
                        "id INTEGER PRIMARY KEY, station TEXT, started_at REAL, last_seen_at REAL, "
                        "status TEXT, uncertain INTEGER, model_id TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS activity_intervals ("
                        "activity_id INTEGER, sequence INTEGER, started_at REAL, ended_at REAL, "
                        "PRIMARY KEY(activity_id,sequence))")
        if recover:
            self.db.execute("UPDATE visits SET status='interrupted' WHERE status='in_progress'")
            self.db.execute("UPDATE activities SET status='interrupted',uncertain=1 WHERE status='in_progress'")
        self.db.commit()
        self.coverage_id = None
        self.last_coverage = None

    def start_visit(self, station, label, visit):
        row_id = self.db.execute(
            "INSERT INTO visits(station,cat_label,entered_at,last_seen_at,status) VALUES(?,?,?,?,?)",
            (station, label, visit.entered_at, visit.last_seen_at, visit.status),
        ).lastrowid
        self.db.commit()
        return row_id

    def update_visit(self, row_id, visit):
        self.db.execute("UPDATE visits SET last_seen_at=?,status=? WHERE id=?",
                        (visit.last_seen_at, visit.status, row_id))
        self.db.commit()

    def mark_coverage(self, timestamp, maximum_gap=10.0):
        if (self.coverage_id is None or self.last_coverage is None or
                timestamp < self.last_coverage or timestamp - self.last_coverage > maximum_gap):
            self.coverage_id = self.db.execute(
                "INSERT INTO coverage(started_at,ended_at) VALUES(?,?)", (timestamp, timestamp)
            ).lastrowid
        else:
            self.db.execute("UPDATE coverage SET ended_at=? WHERE id=?", (timestamp, self.coverage_id))
        self.last_coverage = timestamp
        self.db.commit()

    def break_coverage(self):
        self.coverage_id = self.last_coverage = None

    def summary(self, now, window_seconds=86400):
        cutoff = now - window_seconds
        counts = {}
        durations = {}
        uncertain = {}
        for station, entered, last_seen, status in self.db.execute(
            "SELECT station,entered_at,last_seen_at,status FROM visits "
            "WHERE entered_at<=? AND last_seen_at>=?", (now, cutoff)
        ):
            counts.setdefault(station, 0)
            durations.setdefault(station, 0.0)
            uncertain.setdefault(station, 0)
            if entered >= cutoff:
                counts[station] += 1
                uncertain[station] += int(status in ("uncertain", "interrupted"))
            durations[station] += max(0, min(last_seen, now) - max(entered, cutoff))
        intervals = sorted((max(start, cutoff), min(end, now)) for start, end in self.db.execute(
            "SELECT started_at,ended_at FROM coverage WHERE ended_at>=? AND started_at<=?",
            (cutoff, now),
        ) if min(end, now) > max(start, cutoff))
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return {"counts": counts, "durations": durations, "uncertain": uncertain,
                "observed_seconds": sum(end - start for start, end in merged)}

    def export(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "station", "cat_label", "entered_at_UTC",
                                 "last_seen_at_UTC", "observed_duration_seconds", "status"])
                for row_id, station, label, entered, last_seen, status in self.db.execute(
                    "SELECT * FROM visits ORDER BY id"
                ):
                    # Spreadsheet applications interpret leading formula markers.
                    safe_label = "'" + label if label.startswith(("=", "+", "-", "@")) else label
                    writer.writerow([row_id, station, safe_label, iso_timestamp(entered),
                                     iso_timestamp(last_seen), round(last_seen - entered, 2), status])
            temporary.replace(path)
            return True
        except PermissionError:
            temporary.unlink(missing_ok=True)
            return False

    def close(self):
        self.db.close()

    def resume_activity(self, station, model_id):
        from .behavior import ActivitySession
        row = self.db.execute("SELECT id,started_at,last_seen_at,status,uncertain,model_id "
                              "FROM activities WHERE station=? ORDER BY id DESC LIMIT 1", (station,)).fetchone()
        if not row or row[3] == "completed" or row[5] != model_id:
            return None
        intervals = [list(pair) for pair in self.db.execute(
            "SELECT started_at,ended_at FROM activity_intervals WHERE activity_id=? ORDER BY sequence", (row[0],))]
        return ActivitySession(row[1], row[2], intervals, "in_progress", True, row[0])

    def write_activity(self, station, activity, model_id):
        if activity.row_id is None:
            activity.row_id = self.db.execute("INSERT INTO activities(station,started_at,last_seen_at,status,uncertain,model_id) "
                "VALUES(?,?,?,?,?,?)", (station, activity.started_at, activity.last_seen_at,
                                      activity.status, int(activity.uncertain), model_id)).lastrowid
        else:
            self.db.execute("UPDATE activities SET last_seen_at=?,status=?,uncertain=? WHERE id=?",
                (activity.last_seen_at, activity.status, int(activity.uncertain), activity.row_id))
        for index, (start, end) in enumerate(activity.intervals):
            self.db.execute("INSERT INTO activity_intervals VALUES(?,?,?,?) "
                            "ON CONFLICT(activity_id,sequence) DO UPDATE SET ended_at=excluded.ended_at",
                            (activity.row_id, index, start, end))
        self.db.commit()

    def activity_summary(self, now, window_seconds=86400):
        cutoff = now - window_seconds
        result = {name: {"sessions": 0, "observed_seconds": 0., "incomplete": 0}
                  for name in ("food", "water")}
        for row_id, station, started, last_seen, uncertain, status in self.db.execute(
                "SELECT id,station,started_at,last_seen_at,uncertain,status FROM activities "
                "WHERE started_at<=? AND last_seen_at>=?", (now, cutoff)):
            if station not in result:
                continue
            if started >= cutoff:
                result[station]["sessions"] += 1
                result[station]["incomplete"] += int(bool(uncertain) or status == "interrupted")
            for start, end in self.db.execute("SELECT started_at,ended_at FROM activity_intervals WHERE activity_id=?", (row_id,)):
                result[station]["observed_seconds"] += max(0, min(now, end) - max(cutoff, start))
        return result

    def export_activities(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(["id", "station", "activity", "started_at_UTC", "last_observed_at_UTC",
                                 "observed_activity_seconds", "status", "incomplete", "model_id"])
                for row_id, station, start, end, status, uncertain, model_id in self.db.execute("SELECT * FROM activities ORDER BY id"):
                    observed = self.db.execute("SELECT coalesce(sum(ended_at-started_at),0) FROM activity_intervals "
                                               "WHERE activity_id=?", (row_id,)).fetchone()[0]
                    writer.writerow([row_id, station, "probable_eating" if station == "food" else "probable_drinking",
                                     iso_timestamp(start), iso_timestamp(end), round(observed, 2), status, uncertain, model_id])
            temporary.replace(path)
            return True
        except PermissionError:
            temporary.unlink(missing_ok=True)
            return False
