import unittest
from catcare.tracking import CatTrackCache, tracker_settings


class TrackCacheTests(unittest.TestCase):
    def test_brief_loss_holds_same_id_without_refreshing_observation_time(self):
        cache = CatTrackCache()
        observed = cache.update([(7, (10, 20, 70, 100), .9)], 10)[0]
        self.assertFalse(observed.estimated)
        held = cache.update([], 13)[0]
        self.assertEqual(held.track_id, 7)
        self.assertTrue(held.estimated)
        self.assertEqual(held.last_seen_at, 10)
        resumed = cache.update([(7, (12, 20, 72, 100), .12)], 14)[0]
        self.assertEqual(resumed.track_id, 7)
        self.assertFalse(resumed.estimated)

    def test_lost_box_expires_in_seconds_and_does_not_leave_a_ghost(self):
        cache = CatTrackCache()
        cache.update([(7, (10, 20, 70, 100), .9)], 10)
        self.assertEqual(cache.update([], 16), ())
        cache.update([(7, (10, 20, 70, 100), .9)], 20)
        observed = cache.update([(8, (20, 20, 80, 100), .9)], 21)
        self.assertEqual([track.track_id for track in observed], [8])

    def test_weak_detections_can_recover_existing_tracks_but_cannot_start_them(self):
        settings = tracker_settings(.25)
        self.assertEqual(settings["track_high_thresh"], .25)
        self.assertEqual(settings["new_track_thresh"], .25)
        self.assertLess(settings["track_low_thresh"], settings["new_track_thresh"])
