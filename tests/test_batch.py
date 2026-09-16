import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.inspection.batch import find_image_paths, inspect_images


class TestFindImagePaths(unittest.TestCase):
    def test_find_supported_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)

            (directory / "b.JPG").touch()
            (directory / "a.jpg").touch()
            (directory / "c.png").touch()
            (directory / "memo.txt").touch()

            image_paths = find_image_paths(directory)

            self.assertEqual(
                [path.name for path in image_paths],
                [
                    "a.jpg",
                    "b.JPG",
                    "c.png",
                ],
            )

    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = find_image_paths(temp_dir)

            self.assertEqual(image_paths, [])

    def test_missing_directory_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            find_image_paths("directory_that_does_not_exist")


class TestInspectImages(unittest.TestCase):
    @patch("src.inspection.batch.inspect_image")
    def test_notifications_surround_one_inference_per_image(self, inspect):
        paths = [Path("one.jpg"), Path("two.jpg")]
        events = []
        results = [object(), object()]

        def predict(*, detector, image_path):
            events.append(("infer", image_path))
            return results[paths.index(image_path)]

        inspect.side_effect = predict
        actual = inspect_images(
            object(), paths,
            lambda completed, total, path: events.append(("progress", completed, total, path)),
            image_started_callback=lambda number, total, path: events.append(("start", number, total, path)),
            image_result_callback=lambda number, total, result: events.append(("result", number, total, result)),
        )
        self.assertEqual(actual, results)
        self.assertEqual(events, [
            ("start", 1, 2, paths[0]), ("infer", paths[0]), ("result", 1, 2, results[0]), ("progress", 1, 2, paths[0]),
            ("start", 2, 2, paths[1]), ("infer", paths[1]), ("result", 2, 2, results[1]), ("progress", 2, 2, paths[1]),
        ])
        self.assertEqual(inspect.call_count, 2)

    @patch("src.inspection.batch.inspect_image")
    def test_display_failure_preserves_original_exception_and_stops_next_image(self, inspect):
        error = OSError("display output failure")
        progress = []

        def fail(*args):
            raise error

        with self.assertRaises(OSError) as caught:
            inspect_images(object(), [Path("one.jpg"), Path("two.jpg")],
                           lambda *args: progress.append(args), image_result_callback=fail)
        self.assertIs(caught.exception, error)
        inspect.assert_called_once()
        self.assertEqual(progress, [])

    @patch("src.inspection.batch.inspect_image")
    def test_inspect_multiple_images(self, mock_inspect_image):
        detector = object()

        image_paths = [
            Path("image_001.jpg"),
            Path("image_002.jpg"),
        ]

        mock_inspect_image.side_effect = [
            "result_001",
            "result_002",
        ]

        results = inspect_images(
            detector=detector,
            image_paths=image_paths,
        )

        self.assertEqual(
            results,
            [
                "result_001",
                "result_002",
            ],
        )

        self.assertEqual(
            mock_inspect_image.call_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
