"""Regression checks for full-frame selections and resolution-independent text."""
import importlib.util
import types
import unittest
from unittest.mock import MagicMock, patch
from catcare.display import fit_dimensions, normalized_roi, status_view
from catcare.core import pixel_zones
from catcare import monitor


class GeometryTests(unittest.TestCase):
    def test_high_resolution_and_portrait_frames_fit_without_losing_edges(self):
        for shape in ((1088, 1920, 3), (1920, 1088, 3), (720, 3200, 3)):
            with self.subTest(shape=shape):
                width, height = fit_dimensions(shape, (960, 600))
                self.assertLessEqual(width, 960)
                self.assertLessEqual(height, 600)
                self.assertAlmostEqual(width / height, shape[1] / shape[0], delta=.01)

    def test_reduced_image_selection_maps_to_original_camera_coordinates(self):
        zone = normalized_roi((720, 408, 240, 136), (544, 960, 3))
        self.assertEqual(pixel_zones({"food": zone}, 1920, 1088)["food"], (1440, 816, 1920, 1088))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is needed for display checks")
class DisplayTests(unittest.TestCase):
    def fake_cv(self):
        import numpy as np
        cv = types.SimpleNamespace(WINDOW_AUTOSIZE=1, FONT_HERSHEY_SIMPLEX=0,
                                   LINE_AA=16, INTER_AREA=3, INTER_CUBIC=2)
        for name in ("namedWindow", "moveWindow", "destroyWindow", "putText"):
            setattr(cv, name, MagicMock())
        cv.resize = MagicMock(side_effect=lambda image, size, **kwargs:
                             np.zeros((size[1], size[0], 3), dtype=np.uint8))
        cv.getTextSize = lambda text, font, scale, thickness: ((round(len(text) * 11 * scale), 18), 6)
        return cv

    def test_selector_receives_full_fitted_frame_and_keeps_far_edge_zones(self):
        import numpy as np
        frame = np.zeros((1088, 1920, 3), dtype=np.uint8)
        cv = self.fake_cv()
        cv.selectROI = MagicMock(side_effect=[(0, 0, 240, 136), (720, 0, 240, 136), (0, 408, 240, 136)])
        with patch.object(monitor, "screen_bounds", return_value=(960, 600)):
            zones = monitor.select_zones(frame, cv)
        self.assertIs(cv.resize.call_args.args[0], frame)
        for call in cv.selectROI.call_args_list:
            self.assertEqual(call.args[1].shape, (544, 960, 3))
        self.assertEqual(zones["food"], [.75, 0, 1, .25])
        self.assertEqual(zones["water"], [0, .75, .25, 1])

    def test_status_text_has_same_visible_size_for_small_and_full_hd_video(self):
        import numpy as np
        calls = []
        for shape in ((200, 300, 3), (1088, 1920, 3)):
            cv = self.fake_cv()
            canvas = status_view(np.zeros(shape, dtype=np.uint8), ["Mila | LAST 24 HOURS"], cv, (960, 700))
            self.assertEqual(canvas.shape, (700, 960, 3))
            calls.append(cv.putText.call_args.args[1:])
            self.assertTrue(all(call.args[0].shape == shape for call in cv.resize.call_args_list))
        self.assertEqual(calls[0], calls[1])

    def test_long_status_message_stays_inside_panel(self):
        import numpy as np
        cv = self.fake_cv()
        message = "Visits are station presence, not confirmed eating, drinking, or elimination"
        status_view(np.zeros((1088, 1920, 3), dtype=np.uint8), [message], cv, (480, 700))
        drawn = [call.args[1] for call in cv.putText.call_args_list]
        self.assertEqual(" ".join(drawn), message)
        for text in drawn:
            self.assertLessEqual(cv.getTextSize(text, 0, .65, 2)[0][0], 448)
