"""Learning-image removal preserves labels and excludes trash from new datasets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catcare.dataset import prepare_dataset
from catcare.web_data import ImageLibrary
from catcare.web_runtime import WebRuntime


class ImageCleanupTests(unittest.TestCase):
    def make_library(self, folder):
        library = ImageLibrary(Path(folder) / "sessions")
        session = library.create_session("One real occasion")
        image = library.add_image(session["id"], b"original-private-image", 1920, 1088)
        library.save_label(image["id"], [[.1, .2, .7, .8]])
        return library, session, image

    def test_trash_and_restore_preserve_image_label_and_original_session(self):
        with tempfile.TemporaryDirectory() as folder:
            library, session, image = self.make_library(folder)
            key = image["id"]
            original_image = library.image_path(key).read_bytes()
            original_row = library.images()[0]
            label = library.image_path(key).parent.parent / "labels" / f"{key.split('_')[1]}.txt"
            original_label = label.read_bytes()
            self.assertEqual(library.move_to_trash([key]), 1)
            self.assertEqual(library.images(), [])
            self.assertFalse(label.exists())
            with self.assertRaises(ValueError):
                library.image_path(key)
            self.assertEqual(library.image_path(key, trashed=True).read_bytes(), original_image)
            self.assertEqual(library.trashed_images()[0], original_row)
            library.restore_image(key)
            self.assertEqual(library.images()[0], original_row)
            self.assertEqual(label.read_bytes(), original_label)
            self.assertEqual(library.trashed_images(), [])
            self.assertEqual(library.images()[0]["session_id"], session["id"])

    def test_unreviewed_and_reviewed_empty_labels_stay_distinct_after_restoring(self):
        with tempfile.TemporaryDirectory() as folder:
            library, session, _ = self.make_library(folder)
            unreviewed = library.add_image(session["id"], b"not-reviewed", 640, 480)
            negative = library.add_image(session["id"], b"reviewed-empty", 640, 480)
            library.save_label(negative["id"], [], negative=True)
            for image in (unreviewed, negative):
                library.move_to_trash([image["id"]]); library.restore_image(image["id"])
            rows = {row["id"]: row for row in library.images()}
            self.assertFalse(rows[unreviewed["id"]]["reviewed"])
            self.assertTrue(rows[negative["id"]]["reviewed"])
            self.assertEqual(rows[negative["id"]]["boxes"], [])

    def test_bad_selection_cannot_remove_foreign_active_files_or_models(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _, image = self.make_library(folder)
            key = image["id"]
            model = Path(folder) / "best.pt"; model.write_bytes(b"selected checkpoint")
            missing = 'a' * 32 + '_' + 'b' * 32
            for keys in ([], [key, missing], [key, key], ['../settings']):
                with self.assertRaises(ValueError): library.move_to_trash(keys)
            self.assertTrue(library.image_path(key).is_file())
            with self.assertRaises(ValueError): library.purge_images([key])
            library.move_to_trash([key])
            foreign = ImageLibrary(Path(folder) / "another-profile" / "sessions")
            with self.assertRaises(ValueError): foreign.purge_images([key])
            with self.assertRaises(ValueError): library.purge_images([key, missing])
            self.assertTrue(library.image_path(key, trashed=True).is_file())
            self.assertEqual(library.purge_images([key]), 1)
            self.assertEqual(library.trashed_images(), [])
            self.assertEqual(model.read_bytes(), b"selected checkpoint")

    def test_failed_file_move_rolls_back_the_image_and_its_label(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _, image = self.make_library(folder)
            path = library.image_path(image["id"])
            replace = Path.replace
            def fail_metadata(source, destination):
                if source == path.with_suffix('.json'):
                    raise OSError('simulated storage error')
                return replace(source, destination)
            with patch.object(Path, 'replace', fail_metadata):
                with self.assertRaises(OSError): library.move_to_trash([image["id"]])
            self.assertEqual(library.image_path(image["id"]).read_bytes(), b"original-private-image")
            self.assertTrue(library.images()[0]["reviewed"])
            self.assertEqual(library.trashed_images(), [])

    def test_trash_is_excluded_from_new_prepared_detector_datasets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); library = ImageLibrary(root / "sessions")
            removed = None
            for group in range(3):
                session = library.create_session(f"Independent period {group}")
                for index in range(2):
                    row = library.add_image(session["id"], f"image-{group}-{index}".encode(), 640, 480)
                    library.save_label(row["id"], [[.1, .1, .6, .8]])
                    if group == 0 and index == 0: removed = row
            library.move_to_trash([removed["id"]])
            metadata = prepare_dataset(library.root, root / 'prepared')
            self.assertEqual(sum(row['images'] for row in metadata['splits'].values()), 5)
            self.assertFalse(any(p.read_bytes() == b'image-0-0' for p in (root / 'prepared/images').glob('*/*.jpg')))

    def test_symlink_trash_and_training_jobs_block_image_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            library, _, image = self.make_library(folder)
            outside = Path(folder) / 'outside'; outside.mkdir()
            try:
                (library.root / 'trash').symlink_to(outside, target_is_directory=True)
            except OSError:
                pass
            else:
                with self.assertRaises(ValueError): library.move_to_trash([image['id']])
                self.assertTrue(library.image_path(image['id']).is_file())
            with patch('catcare.config.LOCAL', Path(folder)), patch('catcare.web_runtime.LOCAL', Path(folder)):
                runtime = WebRuntime()
                with patch.object(runtime, 'training_busy', return_value=True):
                    for operation in ('trash', 'restore', 'purge'):
                        with self.assertRaisesRegex(ValueError, 'Stop training'):
                            runtime.manage_images(operation, [image['id']])
                    with self.assertRaisesRegex(ValueError, 'Stop training'):
                        runtime.label_image(image['id'], [])


if __name__ == '__main__': unittest.main()
