"""Report resets are persistent, profile-scoped, and safe around active writers."""
import csv
import json
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from catcare.behavior import ActivitySession
from catcare.config import DEFAULTS, profile_stem
from catcare.core import Visit
from catcare.storage import Store
from catcare.web_runtime import WebRuntime
from web_app import make_server


def populate(store, label='Mila'):
    now = time.time()
    for station in ('litter_box', 'food', 'water'):
        store.start_visit(station, label, Visit(now - 20, now - 10, 'completed'))
    for station in ('food', 'water'):
        store.write_activity(station, ActivitySession(now - 20, now - 10, [[now - 20, now - 10]], 'interrupted', True), 'test-model')
    store.mark_coverage(now - 20); store.mark_coverage(now - 10)


class ReportResetStorageTests(unittest.TestCase):
    def test_reset_clears_all_history_and_resume_state_and_new_records_start_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reports.sqlite3'
            store = Store(path); populate(store)
            deleted = store.reset_reports()
            self.assertEqual(deleted, {'visits': 3, 'activities': 2, 'coverage_intervals': 1})
            self.assertIsNone(store.coverage_id)
            self.assertIsNone(store.last_coverage)
            store.close()
            store = Store(path)
            try:
                self.assertEqual(store.summary(time.time())['counts'], {})
                self.assertEqual(store.summary(time.time())['observed_seconds'], 0)
                self.assertIsNone(store.resume_activity('food', 'test-model'))
                self.assertEqual(store.db.execute('SELECT count(*) FROM activity_intervals').fetchone()[0], 0)
                now = time.time()
                store.start_visit('water', 'Mila', Visit(now - 2, now - 1, 'completed'))
                store.mark_coverage(now - 2); store.mark_coverage(now - 1)
                self.assertEqual(store.summary(now)['counts'], {'water': 1})
                self.assertEqual(store.summary(now)['observed_seconds'], 1)
            finally:
                store.close()

    def test_database_failure_rolls_back_the_whole_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'reports.sqlite3'); populate(store)
            try:
                store.db.execute("CREATE TRIGGER reject_reset BEFORE DELETE ON coverage BEGIN SELECT RAISE(ABORT, 'test failure'); END")
                store.db.commit()
                with self.assertRaises(sqlite3.IntegrityError): store.reset_reports()
                for table, expected in [('visits', 3), ('activities', 2), ('activity_intervals', 2), ('coverage', 1)]:
                    self.assertEqual(store.db.execute(f'SELECT count(*) FROM {table}').fetchone()[0], expected)
                self.assertIsNotNone(store.coverage_id)
            finally:
                store.close()


class ReportResetApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ('catcare.config.LOCAL', 'catcare.web_runtime.LOCAL'):
            item = patch(target, self.root); item.start(); self.addCleanup(item.stop)
        self.runtime = WebRuntime()
        self.runtime.config = {**DEFAULTS, 'cat_label': 'Mila', 'zones': {'water': [.5, .5, .9, .9]}}
        self.server = make_server(self.runtime, 0, training_mode=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), self.thread.join(3), self.runtime.close()))
        _, page = self.request('GET', '/', authenticated=False)
        self.assertIn(b'id="reset-reports"', page)
        self.token = re.search(rb'name="catcare-engine" content="([^\"]+)"', page).group(1).decode()
        self.seed('Mila'); self.seed('Other profile')

    def paths(self, label):
        stem = profile_stem(label)
        return (self.root / 'data' / f'visits_{stem}.sqlite3', self.root / 'reports' / f'visits_{stem}.csv',
                self.root / 'reports' / f'activities_{stem}.csv')

    def seed(self, label):
        path, visits, activities = self.paths(label)
        store = Store(path); populate(store, label)
        store.export(visits); store.export_activities(activities); store.close()

    def request(self, method, path, data=None, authenticated=True, origin=None):
        headers = {'X-CatCare-Token': self.token} if authenticated else {}
        if data is not None: headers['Content-Type'] = 'application/json'
        if origin: headers['Origin'] = origin
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path, json.dumps(data) if data is not None else None, headers)
        response = connection.getresponse(); result = response.status, response.read(); connection.close()
        return result

    def count(self, label):
        store = Store(self.paths(label)[0], recover=False)
        try: return store.db.execute('SELECT count(*) FROM visits').fetchone()[0]
        finally: store.close()

    def test_confirmed_reset_works_in_normal_mode_and_preserves_other_profiles_and_training(self):
        private = self.runtime.profile / 'sessions' / 'fixture.jpg'
        private.parent.mkdir(parents=True); private.write_bytes(b'private training image')
        model = self.root / 'models' / 'custom.pt'; model.parent.mkdir(); model.write_bytes(b'trained weights')
        other_reports = [path.read_bytes() for path in self.paths('Other profile')[1:]]
        code, body = self.request('POST', '/api/reports/reset', {'confirmed': True, 'cat_label': 'Mila'})
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)['exports_ready'])
        self.assertEqual(self.count('Mila'), 0); self.assertEqual(self.count('Other profile'), 3)
        for route in ('/api/report.csv', '/api/behavior/report.csv'):
            status, csv_body = self.request('GET', route)
            self.assertEqual(status, 200)
            self.assertEqual(len(list(csv.reader(csv_body.decode('utf-8-sig').splitlines()))), 1)
        row = json.loads(self.request('GET', '/api/status')[1])
        self.assertEqual(row['summary']['counts'], {})
        self.assertEqual(row['behavior']['summary']['water']['sessions'], 0)
        self.assertEqual(private.read_bytes(), b'private training image')
        self.assertEqual(model.read_bytes(), b'trained weights')
        self.assertEqual([path.read_bytes() for path in self.paths('Other profile')[1:]], other_reports)

    def test_missing_auth_confirmation_wrong_profile_and_live_camera_cannot_erase_history(self):
        valid = {'confirmed': True, 'cat_label': 'Mila'}
        self.assertEqual(self.request('POST', '/api/reports/reset', valid, authenticated=False)[0], 403)
        self.assertEqual(self.request('POST', '/api/reports/reset', valid, origin='https://example.org')[0], 403)
        for confirmed in (None, False, 'true', 1):
            self.assertEqual(self.request('POST', '/api/reports/reset', {'confirmed': confirmed, 'cat_label': 'Mila'})[0], 400)
        self.assertEqual(self.request('POST', '/api/reports/reset', {'confirmed': True, 'cat_label': 'Other profile'})[0], 400)
        with patch.object(self.runtime, 'active', return_value=True):
            self.assertEqual(self.request('POST', '/api/reports/reset', valid)[0], 400)
        self.assertEqual(self.count('Mila'), 3); self.assertEqual(self.count('Other profile'), 3)

    def test_locked_second_csv_restores_first_csv_and_keeps_database_history(self):
        _, visits, activities = self.paths('Mila')
        original = {path: path.read_bytes() for path in (visits, activities)}
        replace = Path.replace
        def locked(source, destination):
            if source == activities: raise PermissionError('spreadsheet lock')
            return replace(source, destination)
        with patch.object(Path, 'replace', locked):
            with self.assertRaisesRegex(ValueError, 'Close the profile'):
                self.runtime.reset_reports('Mila', True)
        self.assertEqual(self.count('Mila'), 3)
        self.assertEqual({path: path.read_bytes() for path in (visits, activities)}, original)

    def test_failed_new_export_never_restores_stale_csv_history_after_reset(self):
        with patch.object(Store, 'export', return_value=False), patch.object(Store, 'export_activities', return_value=False):
            row = self.runtime.reset_reports('Mila', True)
        self.assertFalse(row['exports_ready'])
        self.assertEqual(self.count('Mila'), 0)
        self.assertFalse(any(path.exists() for path in self.paths('Mila')[1:]))
        self.assertEqual(self.runtime.state['summary']['observed_seconds'], 0)


if __name__ == '__main__': unittest.main()
