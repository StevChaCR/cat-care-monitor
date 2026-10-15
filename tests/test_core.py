import unittest
from catcare.core import VisitCounter, assign_station, box_is_clear_of_zone, station_presence, validate_zones


class CounterTests(unittest.TestCase):
    def armed(self):
        counter = VisitCounter()
        counter.observe(False, 0)
        counter.observe(False, 8)
        return counter

    def test_visit_and_short_detection_dropout(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(False, 10.7)
        counter.observe(True, 11.2)
        event, visit = counter.observe(True, 12)[0]
        self.assertEqual((event, visit.entered_at), ("start", 10))
        counter.observe(False, 13)
        self.assertEqual(counter.observe(True, 16), [])
        counter.observe(False, 17)
        event, visit = counter.observe(False, 25)[0]
        self.assertEqual((event, visit.status, visit.last_seen_at), ("end", "completed", 16))

    def test_brief_pass_does_not_start_visit(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(False, 11)
        counter.observe(False, 13)
        self.assertIsNone(counter.visit)
        counter.observe(True, 14)
        counter.observe(True, 15)
        event, visit = counter.observe(True, 16)[0]
        self.assertEqual(visit.entered_at, 14)

    def test_startup_occupied_requires_empty_zone(self):
        counter = VisitCounter()
        self.assertEqual(counter.observe(True, 0), [])
        self.assertEqual(counter.observe(True, 100), [])
        self.assertIsNone(counter.visit)
        counter.observe(False, 101)
        counter.observe(False, 109)
        counter.observe(True, 110)
        self.assertEqual(counter.observe(True, 112)[0][0], "start")

    def test_ambiguous_presence_keeps_same_visit_and_marks_it_uncertain(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(True, 12)
        counter.observe(None, 13)
        visit = counter.visit
        self.assertEqual(counter.observe(None, 15), [])
        self.assertIs(counter.visit, visit)
        self.assertEqual(visit.status, "uncertain")
        self.assertEqual(visit.last_seen_at, 12)
        self.assertEqual(counter.observe(True, 16), [])
        self.assertIs(counter.visit, visit)
        self.assertEqual(visit.last_seen_at, 16)

    def test_long_detector_loss_and_reappearance_do_not_create_extra_visit(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(True, 12)
        first = counter.visit
        for timestamp in (13, 20, 30, 40):
            self.assertEqual(counter.observe(None, timestamp), [])
        self.assertEqual(first.last_seen_at, 12)
        self.assertEqual(counter.observe(True, 42), [])
        self.assertIs(counter.visit, first)
        self.assertEqual(first.last_seen_at, 42)
        counter.observe(False, 43)
        self.assertEqual(counter.observe(False, 46)[0][0], "end")
        counter.observe(True, 50)
        self.assertEqual(counter.observe(True, 52)[0][0], "start")

    def test_unknown_presence_cannot_complete_exit_confirmation(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(True, 12)
        counter.observe(False, 13)
        counter.observe(None, 20)
        self.assertIsNotNone(counter.visit)
        self.assertEqual(counter.observe(False, 21), [])
        self.assertEqual(counter.observe(False, 24)[0][0], "end")

    def test_interruption_requires_observed_departure_before_another_visit(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(True, 12)
        counter.interrupt()
        self.assertTrue(counter.requires_departure)
        for timestamp in (20, 30, 40):
            self.assertEqual(counter.observe(None, timestamp), [])
        self.assertEqual(counter.observe(True, 42), [])
        counter.observe(False, 43)
        counter.observe(False, 46)
        self.assertFalse(counter.requires_departure)
        counter.observe(True, 47)
        self.assertEqual(counter.observe(True, 49)[0][0], "start")

    def test_interruption_does_not_duplicate_a_visit(self):
        counter = self.armed()
        counter.observe(True, 10)
        counter.observe(True, 12)
        self.assertEqual(counter.interrupt().status, "interrupted")
        self.assertEqual(counter.observe(True, 100), [])


class GeometryTests(unittest.TestCase):
    def test_near_boundary_jitter_is_not_a_confirmed_departure(self):
        self.assertFalse(box_is_clear_of_zone((80, 80, 120, 120), (100, 100, 200, 200)))
        self.assertFalse(box_is_clear_of_zone((60, 110, 90, 150), (100, 100, 200, 200)))
        self.assertTrue(box_is_clear_of_zone((0, 0, 50, 50), (100, 100, 200, 200)))

    def test_missing_detection_becomes_unknown_for_an_active_visit(self):
        counter = VisitCounter()
        rectangles = {"water": (100, 100, 200, 200)}
        self.assertIs(station_presence(counter, "water", [], rectangles, None, set()), False)
        counter.visit = type("Visit", (), {})()
        self.assertIsNone(station_presence(counter, "water", [], rectangles, None, set()))

    def test_overlapping_configuration_rejected(self):
        with self.assertRaises(ValueError):
            validate_zones({"food": [.1,.1,.4,.4], "water": [.3,.3,.6,.6]})

    def test_shared_bounding_box_is_ambiguous(self):
        station, ambiguous = assign_station((0,0,100,100), {
            "food": (0,0,50,50), "water": (50,0,100,50),
        })
        self.assertIsNone(station)
        self.assertEqual(ambiguous, {"food", "water"})

    def test_litter_box_requires_lower_body_inside(self):
        self.assertIsNone(assign_station((0,0,100,100), {"litter_box": (0,0,100,40)})[0])
        self.assertEqual(assign_station((0,0,100,100), {"litter_box": (0,70,100,100)})[0], "litter_box")


if __name__ == "__main__":
    unittest.main()
