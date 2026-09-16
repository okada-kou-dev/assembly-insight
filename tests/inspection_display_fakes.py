"""段階表示のテスト用検出結果。実モデルの性能評価には使用しない。"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

NAMES = {0: "bolt", 1: "washer", 2: "spacer", 3: "cap_nut", 4: "other"}
NORMAL_CLASSES = ["washer", "bolt", "cap_nut", "washer", "spacer"]


def make_result(classes=None, *, coordinates=None):
    classes = NORMAL_CLASSES if classes is None else classes
    boxes = []
    for index, name in enumerate(classes):
        xyxy = coordinates[index] if coordinates is not None else [20 + index * 70, 100, 65 + index * 70, 170]
        boxes.append(SimpleNamespace(
            cls=np.array([next(key for key, value in NAMES.items() if value == name)]),
            conf=np.array([0.91 + index * 0.01]),
            xyxy=np.array([xyxy], dtype=float),
        ))
    original = np.full((300, 500, 3), (30, 80, 170), dtype=np.uint8)
    return SimpleNamespace(boxes=boxes, names=NAMES.copy(), orig_img=original)


def make_position_result(washers=(70, 150)):
    classes = ["bolt", "cap_nut", "spacer"] + ["washer"] * len(washers)
    return make_result(classes, coordinates=[[x - 5, 45, x + 5, 55] for x in (10, 210, 110, *washers)])


def write_input_images(directory, count=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"20260901_08000{index}_{index + 1:03d}.png"
        encoded, data = cv2.imencode(".png", make_result([]).orig_img)
        if not encoded:
            raise RuntimeError("テスト画像の生成に失敗しました。")
        data.tofile(path)
        paths.append(path)
    return paths


class FakeDetector:
    def __init__(self, results=None, fail_at=None):
        self.results = results if results is not None else [
            make_result(), make_result(["bolt", "washer", "spacer", "cap_nut"]), make_result([]),
        ]
        self.fail_at = fail_at
        self.calls = []

    def predict(self, image_path):
        self.calls.append(Path(image_path))
        index = len(self.calls) - 1
        if index == self.fail_at:
            raise RuntimeError("simulated inference failure")
        return self.results[index]


class RecordingPanel:
    def __init__(self):
        self.events = []
        self.progress_events = []

    def show_progress(self, state):
        self.progress_events.append((state.inspected, state.total))

    def show(self, state):
        self.events.append({
            "stage": state.stage, "path": state.active_path,
            "decision": state.decision, "quantities": list(state.quantity_rows),
            "detections": list(state.detection_rows), "inspected": state.inspected,
            "saved": state.saved, "saving_path": state.saving_path,
        })

    show_detection_frame = show
