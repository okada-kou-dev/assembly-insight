from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspection_rules import InspectionDecision, judge_part_counts
from src.inspection.position import WasherPosition, analyze_washer_position
from src.vision.detector import PartDetector


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float] | None = None


@dataclass
class ImageInspection:
    image_path: Path
    detections: list[Detection]
    counts: dict[str, int]
    decision: InspectionDecision
    model_result: Any
    washer_position: WasherPosition | None = None


def inspect_image(
    detector: PartDetector,
    image_path: str | Path,
) -> ImageInspection:
    """
    画像1枚を推論し、数量判定と位置照合を返す。

    単一ワッシャ欠品で側が明確な場合だけ欠品種類を細分化する。
    位置が未確定の場合や、他の数量異常の判定は変更しない。
    """
    image_path = Path(image_path)

    result = detector.predict(image_path)

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        detections.append(
            Detection(
                class_name=class_name,
                confidence=confidence,
                xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
            )
        )

    detected_classes = [detection.class_name for detection in detections]

    counter = Counter(detected_classes)

    counts = {
        "bolt": counter["bolt"],
        "washer": counter["washer"],
        "spacer": counter["spacer"],
        "cap_nut": counter["cap_nut"],
    }

    decision = judge_part_counts(counts)
    washer_position = analyze_washer_position(detections)
    if decision.defect_type == "washer_missing" and washer_position.missing_defect_type is not None:
        decision = InspectionDecision(
            inspection_result=decision.inspection_result,
            defect_type=washer_position.missing_defect_type,
            counts=decision.counts,
        )

    return ImageInspection(
        image_path=image_path,
        detections=detections,
        counts=counts,
        decision=decision,
        model_result=result,
        washer_position=washer_position,
    )
