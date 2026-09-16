from pathlib import Path

from src.inspection.metadata import parse_captured_at
from src.inspection.service import ImageInspection
from src.storage.database import save_inspection


def get_representative_confidence(
    inspection: ImageInspection,
) -> float | None:
    """
    画像1枚の代表confidenceを返す。

    MVPでは、検出された全物体のconfidenceの最小値を使用する。
    検出が0件の場合はNoneを返す。
    """
    if not inspection.detections:
        return None

    return min(detection.confidence for detection in inspection.detections)


def save_image_inspection(
    db_path: str | Path,
    inspection: ImageInspection,
) -> int:
    """
    ImageInspectionをSQLiteへ保存する。
    """
    captured_at = parse_captured_at(inspection.image_path)

    confidence = get_representative_confidence(inspection)

    return save_inspection(
        db_path=db_path,
        image_path=inspection.image_path,
        captured_at=captured_at,
        inspection_result=(inspection.decision.inspection_result),
        defect_type=(inspection.decision.defect_type),
        confidence=confidence,
    )
