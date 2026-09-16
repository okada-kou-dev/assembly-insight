"""全画像共通の信頼度と重複抑制。正解・ファイル名・期待個数を使用しない。"""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from src.vision.detector import PartDetector

CLASSES = ('bolt', 'washer', 'spacer', 'cap_nut')


@dataclass(frozen=True)
class DetectorSettings:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    raw_conf: float = 0.01
    raw_iou: float = 0.95
    class_conf: tuple[float, ...] | None = None

    def __post_init__(self):
        thresholds = self.class_conf if self.class_conf is not None else (self.conf,) * 4
        if (len(thresholds) != 4 or not all(isfinite(v) and self.raw_conf <= v <= 1 for v in thresholds)
                or not 0 < self.raw_conf <= 1 or not 0 < self.iou <= self.raw_iou <= 1
                or self.imgsz not in (640, 960, 1280)):
            raise ValueError('invalid detector thresholds or image size')

    def to_dict(self):
        return asdict(self)


def overlap_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def selected_indices(boxes, settings):
    """boxesは[x1,y1,x2,y2,confidence,class_id]。クラス別の通常NMSを適用。"""
    thresholds = settings.class_conf or (settings.conf,) * 4
    for box in boxes:
        if (len(box) != 6 or not all(isfinite(v) for v in box) or int(box[5]) != box[5]
                or not 0 <= int(box[5]) < 4 or not 0 <= box[4] <= 1
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('invalid detector output box')
    remaining = sorted((i for i, box in enumerate(boxes) if box[4] >= thresholds[int(box[5])]),
                       key=lambda i: (-boxes[i][4], i))
    selected = []
    for index in remaining:
        candidate = boxes[index]
        if any(candidate[5] == boxes[prior][5] and overlap_iou(candidate, boxes[prior]) > settings.iou
               for prior in selected):
            continue
        selected.append(index)
    return selected


class FilteredPartDetector(PartDetector):
    def __init__(self, model_path, settings=None):
        self.settings = settings or DetectorSettings()
        super().__init__(model_path=model_path, conf=self.settings.conf)
        if tuple(self.model.names[i] for i in range(len(self.model.names))) != CLASSES:
            raise ValueError('model class mapping differs')
        self.last_raw = None

    def predict(self, image_path):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = self.model.predict(source=str(path), conf=self.settings.raw_conf, iou=self.settings.raw_iou,
                                 imgsz=self.settings.imgsz, end2end=False, verbose=False)[0]
        self.last_raw = raw
        indices = selected_indices(raw.boxes.data.tolist(), self.settings)
        result = raw.new()
        result.update(boxes=raw.boxes.data[indices])
        return result
