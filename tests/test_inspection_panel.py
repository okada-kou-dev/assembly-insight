import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from inspection_display_fakes import FakeDetector, RecordingPanel, make_result, write_input_images
from src.inspection.service import inspect_image
from src.inspection.workflow import inspect_directory_and_save
from src.storage.database import list_inspections
from src.ui.inspection_animation import SPEED_PRESETS, render_detection_frame
from src.ui.inspection_panel import BatchDisplayState, BatchPresenter


class TestBatchPresenter(unittest.TestCase):
    def test_all_speeds_keep_inference_decisions_and_saved_content_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_input_images(root / "images")
            saved_contents = []
            saved_previews = []
            for index, speed_name in enumerate(SPEED_PRESETS):
                detector, panel, wait = FakeDetector(), RecordingPanel(), Mock()
                db_path = root / f"run_{index}.db"
                state = BatchDisplayState(str(db_path), speed_name)
                presenter = BatchPresenter(state, panel, root / f"previews_{index}", wait=wait)
                with patch("src.llm.client.urlopen") as llm, \
                     patch("src.ui.inspection_panel.render_detection_frame", wraps=render_detection_frame) as render_frame:
                    inspections = inspect_directory_and_save(
                        detector, root / "images", db_path, presenter.progress,
                        image_started_callback=presenter.image_started,
                        image_result_callback=presenter.image_result,
                        save_progress_callback=presenter.save_progress,
                    )
                    llm.assert_not_called()
                    if speed_name == "結果のみ":
                        self.assertEqual(render_frame.call_count, len(paths))
                self.assertEqual(detector.calls, paths)
                self.assertEqual([item.decision.inspection_result for item in inspections], ["OK", "NG", "NG"])
                self.assertEqual([item.decision.defect_type for item in inspections], [None, "washer_after_spacer_missing", "multiple_missing"])
                self.assertEqual(state.saved, 3)
                self.assertEqual(len(state.previews), 3)
                self.assertEqual(len(list((root / f"previews_{index}").glob("*.jpg"))), 3)
                saved_contents.append(list_inspections(db_path))
                saved_previews.append([Path(state.previews[str(path)]).read_bytes() for path in paths])
                if speed_name == "結果のみ":
                    wait.assert_not_called()
                    self.assertEqual(panel.events, [])
                    self.assertEqual(panel.progress_events, [(0, 3), (1, 3), (2, 3), (3, 3)])
                    self.assertIsNone(state.frame)
                else:
                    self.assertEqual(wait.call_count, 9 + 3)
                    self.assertTrue(any("検出結果を順次表示中" in event["stage"] for event in panel.events))
                for event in panel.events:
                    if "検出結果を順次表示中" in event["stage"] or "演出なし" in event["stage"]:
                        self.assertIsNone(event["decision"])
                        self.assertEqual(event["quantities"], [])
                    self.assertNotIn("表示演出", event["stage"])
            self.assertEqual(saved_contents[0], saved_contents[1])
            self.assertEqual(saved_previews[0], saved_previews[1])
            self.assertEqual(len(saved_contents), 2)

    def test_results_only_progress_stops_at_last_completed_image_on_inference_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_input_images(root / "images")
            detector, panel = FakeDetector(fail_at=1), RecordingPanel()
            state = BatchDisplayState(str(root / "failed.db"), "結果のみ")
            presenter = BatchPresenter(state, panel, root / "previews", wait=Mock())
            with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
                inspect_directory_and_save(
                    detector, root / "images", state.db_path, presenter.progress,
                    image_started_callback=presenter.image_started,
                    image_result_callback=presenter.image_result,
                    save_progress_callback=presenter.save_progress,
                )
            self.assertEqual(panel.progress_events, [(0, 3), (1, 3)])
            self.assertEqual(panel.events, [])
            self.assertEqual(detector.calls, paths[:2])
            self.assertFalse(Path(state.db_path).exists())

    def test_next_image_clears_previous_frames_decision_and_counts_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_input_images(root / "images", 2)
            state = BatchDisplayState(str(root / "new.db"), "結果のみ")
            panel = RecordingPanel()
            presenter = BatchPresenter(state, panel, root / "previews", wait=Mock())
            presenter.image_started(1, 2, paths[0])
            presenter.image_result(1, 2, inspect_image(FakeDetector([make_result()]), paths[0]))
            self.assertEqual(state.decision, "OK")
            presenter.image_started(2, 2, paths[1])
            self.assertIsNone(state.decision)
            self.assertEqual(state.quantity_rows, [])
            self.assertEqual(state.detection_rows, [])
            self.assertEqual(state.stage, "画像AI推論中")
            self.assertEqual(state.inspected, 1)
            self.assertEqual(state.saved, 0)


if __name__ == "__main__":
    unittest.main()
