import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.inspection.workflow import (
    inspect_directory_and_save,
)


class TestInspectionWorkflow(unittest.TestCase):
    @patch("src.inspection.workflow.save_image_inspection")
    @patch("src.inspection.workflow.inspect_images")
    def test_optional_callbacks_and_save_notifications(self, inspect, save):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            paths = [image_dir / "20260901_080000_001.jpg", image_dir / "20260901_090000_002.jpg"]
            for path in paths:
                path.touch()
            inspect.return_value = [SimpleNamespace(image_path=path) for path in paths]
            starts, results, progress = object(), object(), object()
            events = []
            save.side_effect = lambda **kwargs: events.append(("write", kwargs["inspection"].image_path))
            inspect_directory_and_save(
                object(), image_dir, image_dir / "new.db", progress,
                image_started_callback=starts, image_result_callback=results,
                save_progress_callback=lambda count, total, path: events.append(("notify", count, total, path)),
            )
            self.assertIs(inspect.call_args.kwargs["progress_callback"], progress)
            self.assertIs(inspect.call_args.kwargs["image_started_callback"], starts)
            self.assertIs(inspect.call_args.kwargs["image_result_callback"], results)
            self.assertEqual(events, [
                ("notify", 0, 2, paths[0]), ("write", paths[0]), ("notify", 1, 2, paths[0]),
                ("notify", 1, 2, paths[1]), ("write", paths[1]), ("notify", 2, 2, paths[1]),
            ])

    @patch("src.inspection.workflow.save_image_inspection")
    @patch("src.inspection.workflow.inspect_images")
    def test_failed_save_reports_current_image_without_claiming_success(self, inspect, save):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            paths = [image_dir / f"20260901_08000{i}_{i}.jpg" for i in range(3)]
            for path in paths:
                path.touch()
            inspect.return_value = [SimpleNamespace(image_path=path) for path in paths]
            error = OSError("simulated database failure")
            save.side_effect = [1, error]
            events = []
            with self.assertRaises(OSError) as caught:
                inspect_directory_and_save(object(), image_dir, image_dir / "new.db",
                                           save_progress_callback=lambda *event: events.append(event))
            self.assertIs(caught.exception, error)
            self.assertEqual(save.call_count, 2)
            self.assertEqual(events[-1], (1, 3, paths[1]))

    @patch("src.inspection.workflow.save_image_inspection")
    @patch("src.inspection.workflow.inspect_images")
    def test_inspect_directory_and_save(
        self,
        mock_inspect_images,
        mock_save_image_inspection,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)

            (image_dir / "20260901_080000_001.jpg").touch()

            (image_dir / "20260901_100000_002.jpg").touch()

            fake_inspections = [
                object(),
                object(),
            ]

            mock_inspect_images.return_value = fake_inspections

            detector = object()
            db_path = image_dir / "inspection.db"

            result = inspect_directory_and_save(
                detector=detector,
                image_dir=image_dir,
                db_path=db_path,
            )

            self.assertEqual(
                result,
                fake_inspections,
            )

            self.assertEqual(
                mock_save_image_inspection.call_count,
                2,
            )

    @patch("src.inspection.workflow.inspect_images")
    def test_invalid_filename_fails_before_inference(
        self,
        mock_inspect_images,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)

            (image_dir / "normal_test_001.jpg").touch()

            with self.assertRaises(ValueError):
                inspect_directory_and_save(
                    detector=object(),
                    image_dir=image_dir,
                    db_path=image_dir / "inspection.db",
                )

            mock_inspect_images.assert_not_called()

    def test_empty_directory_raises_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                inspect_directory_and_save(
                    detector=object(),
                    image_dir=temp_dir,
                    db_path=(Path(temp_dir) / "inspection.db"),
                )


if __name__ == "__main__":
    unittest.main()
