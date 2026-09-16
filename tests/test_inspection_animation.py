import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from inspection_display_fakes import make_result, write_input_images
from src.ui.inspection_animation import (
    FRAME_HEIGHT, FRAME_WIDTH, SPEED_PRESETS, hold_decision,
    load_original_image, ordered_detections, play_detection_steps,
    quantity_comparison, render_detection_frame,
)


class TestInspectionAnimation(unittest.TestCase):
    def test_class_position_and_source_index_define_stable_order(self):
        result = make_result(
            ["washer", "washer", "spacer", "bolt", "cap_nut", "washer"],
            coordinates=[[100, 40, 120, 60], [20, 80, 40, 100], [5, 5, 10, 10],
                         [10, 10, 20, 20], [20, 20, 30, 30], [20, 80, 40, 100]],
        )
        ordered = ordered_detections(result)
        self.assertEqual([item.source_index for item in ordered], [4, 3, 2, 1, 5, 0])
        self.assertEqual([item.label for item in ordered[-3:]], ["ワッシャ1", "ワッシャ2", "ワッシャ3"])
        self.assertFalse(any("前" in item.label or "後" in item.label for item in ordered))

    def test_frames_add_one_actual_box_and_keep_input_unchanged(self):
        result = make_result()
        original = result.orig_img.copy()
        coordinates = [box.xyxy.copy() for box in result.boxes]
        displayed = []
        wait = Mock()
        final = play_detection_steps(result, SPEED_PRESETS["ステップ表示"],
                                     lambda frame, items, total: displayed.append((frame, items, total)), wait=wait)
        self.assertEqual([len(items) for _, items, _ in displayed], [1, 2, 3, 4, 5])
        self.assertEqual({item.source_index for item in final}, set(range(5)))
        self.assertEqual(displayed[-1][1], final)
        self.assertTrue(all(total == 5 for _, _, total in displayed))
        self.assertEqual([call.args[0] for call in wait.call_args_list], [0.1] * 5)
        self.assertTrue(all(frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3) for frame, _, _ in displayed))
        self.assertFalse(np.array_equal(displayed[0][0], displayed[-1][0]))
        np.testing.assert_array_equal(result.orig_img, original)
        for box, before in zip(result.boxes, coordinates):
            np.testing.assert_array_equal(box.xyxy, before)

    def test_missing_and_excess_parts_never_add_invented_boxes(self):
        for classes in (["washer"], ["washer"] * 7, ["other", "cap_nut"]):
            with self.subTest(classes=classes):
                result = make_result(classes)
                shown = []
                final = play_detection_steps(result, SPEED_PRESETS["ステップ表示"],
                                             lambda frame, items, total: shown.append(items), wait=Mock())
                self.assertEqual(len(final), len(classes))
                self.assertEqual([len(items) for items in shown], list(range(1, len(classes) + 1)))
                self.assertEqual([item.confidence for item in sorted(final, key=lambda item: item.source_index)],
                                 [float(box.conf.item()) for box in result.boxes])

    def test_no_detections_displays_original_once_without_step_wait(self):
        result = make_result([])
        display, wait = Mock(), Mock()
        self.assertEqual(play_detection_steps(result, SPEED_PRESETS["ステップ表示"], display, wait=wait), ())
        display.assert_called_once()
        self.assertEqual(display.call_args.args[1:], ((), 0))
        np.testing.assert_array_equal(display.call_args.args[0], render_detection_frame(result.orig_img))
        wait.assert_not_called()
        self.assertTrue(all(row["照合"] == "未検出" for row in quantity_comparison(
            {"bolt": 0, "washer": 0, "spacer": 0, "cap_nut": 0})))

    def test_animation_off_has_one_complete_frame_and_no_wait(self):
        display, wait = Mock(), Mock()
        result = make_result()
        final = play_detection_steps(result, SPEED_PRESETS["結果のみ"], display, wait=wait)
        hold_decision(SPEED_PRESETS["結果のみ"], wait=wait)
        display.assert_called_once()
        self.assertEqual(len(display.call_args.args[1]), 5)
        np.testing.assert_array_equal(display.call_args.args[0], render_detection_frame(result.orig_img, final))
        wait.assert_not_called()

    def test_presets_use_separate_decision_hold(self):
        for name, expected in [("ステップ表示", 0.3)]:
            wait = Mock()
            hold_decision(SPEED_PRESETS[name], wait=wait)
            wait.assert_called_once_with(expected)

    def test_frame_keeps_bgr_color_and_image_aspect_ratio(self):
        original = np.full((1200, 600, 3), (11, 44, 199), dtype=np.uint8)
        frame = render_detection_frame(original)
        np.testing.assert_array_equal(frame[300, 480], [11, 44, 199])
        np.testing.assert_array_equal(frame[300, 0], [32, 17, 11])
        self.assertEqual(frame.shape, (600, 800, 3))

    def test_unicode_image_path_is_supported_and_invalid_image_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_input_images(Path(directory) / "検査画像", 1)
            np.testing.assert_array_equal(load_original_image(paths[0]), make_result([]).orig_img)
            bad = Path(directory) / "壊れた画像.png"
            bad.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "画像を読み込めません"):
                load_original_image(bad)

    def test_invalid_boxes_do_not_create_fabricated_frames(self):
        for coordinates in ([[float("nan"), 1, 2, 3]], [[20, 20, 10, 10]]):
            with self.subTest(coordinates=coordinates):
                display = Mock()
                with self.assertRaises(ValueError):
                    play_detection_steps(make_result(["washer"], coordinates=coordinates),
                                         SPEED_PRESETS["ステップ表示"], display, wait=Mock())
                display.assert_not_called()


if __name__ == "__main__":
    unittest.main()
