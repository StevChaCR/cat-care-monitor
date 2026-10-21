"""Actual class/station coverage, seven Other clips, and whole-period splitting."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare.behavior_data import ClipLibrary, behavior_readiness, prepare_behavior_dataset, _random_behavior_split
from catcare.web_training import BehaviorTrainingJob


class BehaviorReadinessTests(unittest.TestCase):
    def make_28_clips(self, root, water_only=False, one_period=False):
        library = ClipLibrary(root / 'sessions')
        sessions = [library.create_session(f'Real occasion {index}') for index in range(1 if one_period else 3)]
        initial = _random_behavior_split([row['id'] for row in sessions], 42)
        rich_period = initial['test'][0]  # Its three Other clips must go to training, not test.
        for group, session in enumerate(sessions):
            rich = session['id'] == rich_period
            counts = (14, 7, 7) if one_period else (6, 3, 3) if rich else (4, 2, 2)
            index = 0
            for label, count in zip(('eating', 'drinking', 'other'), counts):
                for instance in range(count):
                    station = 'water' if label == 'drinking' or (label == 'other' and (water_only or instance % 2)) else 'food'
                    clip = library.add_clip(session['id'], station,
                        tuple(f'{group}-{index}-{frame}'.encode() for frame in range(16)),
                        [group * 86400 + index * 20 + frame * .25 for frame in range(16)], [0, 0, 100, 100])
                    library.label_clip(clip['id'], label); index += 1
        return library, rich_period

    def test_14_eating_7_drinking_7_other_can_train_without_a_lucky_random_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); library, rich = self.make_28_clips(root)
            report = behavior_readiness(library.root)
            self.assertEqual(report['counts'], {'eating': 14, 'drinking': 7, 'other': 7})
            self.assertTrue(report['ready'], report['issues'])
            self.assertEqual(report['split_strategy'], 'class_coverage_by_period')
            dataset = prepare_behavior_dataset(library.root, root / 'data.json')
            self.assertEqual(dataset['splits']['train']['sessions'], [rich])
            self.assertEqual(sorted(row['counts']['other'] for row in dataset['splits'].values()), [2, 2, 3])
            assigned = [set(row['sessions']) for row in dataset['splits'].values()]
            self.assertTrue(all(not a & b for index, a in enumerate(assigned) for b in assigned[index + 1:]))

    def test_seven_other_clips_from_only_water_report_the_missing_food_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _ = self.make_28_clips(Path(folder), water_only=True)
            report = behavior_readiness(library.root)
            self.assertEqual(report['counts']['other'], 7)
            self.assertFalse(report['ready'])
            self.assertTrue(any('Other at food: 0/1' in item for item in report['issues']))
            with self.assertRaises(ValueError): prepare_behavior_dataset(library.root, Path(folder) / 'bad.json')

    def test_empty_session_names_do_not_make_one_recording_three_independent_periods(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _ = self.make_28_clips(Path(folder), one_period=True)
            library.create_session('Empty label A'); library.create_session('Empty label B')
            report = behavior_readiness(library.root)
            self.assertEqual(report['counts'], {'eating': 14, 'drinking': 7, 'other': 7})
            self.assertEqual(report['usable_periods'], 1)
            self.assertFalse(report['ready'])
            self.assertTrue(any('found 1' in issue for issue in report['issues']))
            self.assertEqual(sum(period['split'] is None for period in report['periods']), 2)

    def test_short_single_activity_occasions_can_be_grouped_into_valid_sets(self):
        with tempfile.TemporaryDirectory() as folder:
            library = ClipLibrary(Path(folder) / 'sessions')
            for group in range(9):
                label = ('eating', 'drinking', 'other')[group // 3]
                session = library.create_session(f'Independent occasion {group}')
                for index in range(3):
                    station = 'water' if label == 'drinking' or (label == 'other' and index % 2) else 'food'
                    clip = library.add_clip(session['id'], station, tuple(f'{group}-{index}-{f}'.encode() for f in range(16)),
                        [group * 86400 + index * 20 + .25 * f for f in range(16)], [0, 0, 100, 100])
                    library.label_clip(clip['id'], label)
            report = behavior_readiness(library.root)
            self.assertTrue(report['ready'], report['issues'])
            dataset = prepare_behavior_dataset(library.root, Path(folder) / 'data.json')
            self.assertEqual(sum(len(row['sessions']) for row in dataset['splits'].values()), 9)

    def test_review_and_frame_integrity_are_still_required_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _ = self.make_28_clips(Path(folder))
            key = library.clips()[0]['id']
            library.frame_path(key, 0).unlink()
            report = behavior_readiness(library.root)
            self.assertTrue(report['ready'])
            self.assertFalse(report['integrity_checked'])
            with self.assertRaises(OSError): prepare_behavior_dataset(library.root, Path(folder) / 'data.json')
            with patch('catcare.web_training.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, 'Collect and review'):
                    BehaviorTrainingJob().start(Path(folder), Path(folder) / 'empty', 5, 'cpu')
                process.assert_not_called()


if __name__ == '__main__': unittest.main()
