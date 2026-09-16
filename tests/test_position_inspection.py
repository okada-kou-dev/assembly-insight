import tempfile
import unittest
from pathlib import Path

import numpy as np

from inspection_display_fakes import FakeDetector, make_position_result, make_result
from src.inspection.persistence import get_representative_confidence, save_image_inspection
from src.inspection.service import Detection, inspect_image
from src.storage.database import list_inspections


class TestPositionInspection(unittest.TestCase):
    def test_retains_real_boxes_without_repeating_inference_or_mutating_model_result(self):
        result = make_position_result()
        before = [box.xyxy.copy() for box in result.boxes]
        detector = FakeDetector([result])
        inspection = inspect_image(detector, "input.jpg")
        self.assertEqual(detector.calls, [Path("input.jpg")])
        self.assertIs(inspection.model_result, result)
        self.assertEqual(inspection.decision.inspection_result, "OK")
        self.assertEqual(inspection.washer_position.reason, "both_sides_present")
        for actual, expected, source in zip(inspection.detections, before, result.boxes):
            self.assertEqual(actual.xyxy, tuple(expected[0]))
            np.testing.assert_array_equal(source.xyxy, expected)

    def test_each_missing_side_is_ng_and_survives_database_save(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, (remaining, expected) in enumerate(((70, "washer_after_spacer_missing"), (150, "washer_before_spacer_missing"))):
                with self.subTest(remaining=remaining):
                    result = make_position_result((remaining,))
                    inspection = inspect_image(FakeDetector([result]), f"20260901_08000{index}_001.jpg")
                    self.assertEqual(inspection.decision.inspection_result, "NG")
                    self.assertEqual(inspection.decision.defect_type, expected)
                    self.assertEqual(inspection.counts, {"bolt": 1, "washer": 1, "spacer": 1, "cap_nut": 1})
                    self.assertEqual(get_representative_confidence(inspection), min(float(box.conf.item()) for box in result.boxes))
                    db_path = Path(directory) / f"result_{index}.db"
                    save_image_inspection(db_path, inspection)
                    rows = list_inspections(db_path)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["defect_type"], expected)

    def test_ambiguous_single_washer_preserves_generic_missing(self):
        inspection = inspect_image(FakeDetector([make_position_result((110,))]), "input.jpg")
        self.assertEqual(inspection.washer_position.status, "unresolved")
        self.assertEqual(inspection.decision.defect_type, "washer_missing")

    def test_position_analysis_does_not_replace_other_quantity_decisions(self):
        for classes, expected in ((["bolt", "cap_nut", "washer"], "spacer_missing"), (["bolt", "cap_nut"], "multiple_missing")):
            with self.subTest(classes=classes):
                # spacer_missingのケースはワッシャ2枚を必要とする。
                if expected == "spacer_missing":
                    classes = classes + ["washer"]
                inspection = inspect_image(FakeDetector([make_result(classes)]), "input.jpg")
                self.assertEqual(inspection.decision.defect_type, expected)
        normal_counts = inspect_image(FakeDetector([make_position_result((60, 80))]), "input.jpg")
        self.assertEqual(normal_counts.decision.inspection_result, "OK")
        self.assertEqual(normal_counts.washer_position.reason, "washers_on_same_side")
        excess = inspect_image(FakeDetector([make_position_result((60, 80, 150))]), "input.jpg")
        self.assertEqual(excess.decision.defect_type, "quantity_mismatch")

    def test_legacy_detection_constructor_is_still_supported(self):
        self.assertIsNone(Detection("washer", 0.9).xyxy)


if __name__ == "__main__":
    unittest.main()
