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
        if recover:
            self.db.execute("UPDATE visits SET status='interrupted' WHERE status='in_progress'")
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
