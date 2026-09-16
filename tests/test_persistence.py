import tempfile
import unittest
from pathlib import Path

from inspection_rules import InspectionDecision
from src.inspection.persistence import (
    get_representative_confidence,
    save_image_inspection,
)
from src.inspection.service import (
    Detection,
    ImageInspection,
)
from src.storage.database import list_inspections


class TestPersistence(unittest.TestCase):
    def test_representative_confidence_uses_minimum(self):
        inspection = ImageInspection(
            image_path=Path("20260901_080000_001.jpg"),
            detections=[
                Detection("bolt", 0.959),
                Detection("washer", 0.994),
                Detection("washer", 0.979),
                Detection("spacer", 0.993),
                Detection("cap_nut", 0.941),
            ],
            counts={
                "bolt": 1,
                "washer": 2,
                "spacer": 1,
                "cap_nut": 1,
            },
            decision=InspectionDecision(
                inspection_result="OK",
                defect_type=None,
                counts={
                    "bolt": 1,
                    "washer": 2,
                    "spacer": 1,
                    "cap_nut": 1,
                },
            ),
            model_result=None,
        )

        confidence = get_representative_confidence(inspection)

        self.assertAlmostEqual(
            confidence,
            0.941,
        )

    def test_no_detection_returns_none_confidence(self):
        inspection = ImageInspection(
            image_path=Path("20260901_080000_001.jpg"),
            detections=[],
            counts={
                "bolt": 0,
                "washer": 0,
                "spacer": 0,
                "cap_nut": 0,
            },
            decision=InspectionDecision(
                inspection_result="NG",
                defect_type="multiple_missing",
                counts={
                    "bolt": 0,
                    "washer": 0,
                    "spacer": 0,
                    "cap_nut": 0,
                },
            ),
            model_result=None,
        )

        confidence = get_representative_confidence(inspection)

        self.assertIsNone(confidence)

    def test_save_image_inspection_to_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            inspection = ImageInspection(
                image_path=Path("20260905_130000_004.jpg"),
                detections=[
                    Detection("bolt", 0.974),
                    Detection("washer", 0.994),
                    Detection("spacer", 0.968),
                    Detection("cap_nut", 0.922),
                ],
                counts={
                    "bolt": 1,
                    "washer": 1,
                    "spacer": 1,
                    "cap_nut": 1,
                },
                decision=InspectionDecision(
                    inspection_result="NG",
                    defect_type="washer_missing",
                    counts={
                        "bolt": 1,
                        "washer": 1,
                        "spacer": 1,
                        "cap_nut": 1,
                    },
                ),
                model_result=None,
            )

            image_id = save_image_inspection(
                db_path=db_path,
                inspection=inspection,
            )

            rows = list_inspections(db_path)

            self.assertEqual(image_id, 1)
            self.assertEqual(len(rows), 1)

            row = rows[0]

            self.assertEqual(
                row["image_path"],
                "20260905_130000_004.jpg",
            )

            self.assertEqual(
                row["captured_at"],
                "2026-09-05T13:00:00",
            )

            self.assertEqual(
                row["inspection_result"],
                "NG",
            )

            self.assertEqual(
                row["defect_type"],
                "washer_missing",
            )

            self.assertAlmostEqual(
                row["confidence"],
                0.922,
            )

    def test_invalid_filename_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "inspection.db"

            inspection = ImageInspection(
                image_path=Path("normal_test_001.jpg"),
                detections=[],
                counts={
                    "bolt": 1,
                    "washer": 2,
                    "spacer": 1,
                    "cap_nut": 1,
                },
                decision=InspectionDecision(
                    inspection_result="OK",
                    defect_type=None,
                    counts={
                        "bolt": 1,
                        "washer": 2,
                        "spacer": 1,
                        "cap_nut": 1,
                    },
                ),
                model_result=None,
            )

            with self.assertRaises(ValueError):
                save_image_inspection(
                    db_path=db_path,
                    inspection=inspection,
                )


if __name__ == "__main__":
    unittest.main()
