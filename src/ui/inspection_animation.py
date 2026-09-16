"""実検出結果の表示専用処理。推論・判定・DB保存は行わない。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import sleep
from typing import Any

import cv2
import numpy as np

from inspection_rules import EXPECTED_COUNTS

PART_LABELS = {
    "cap_nut": "化粧ナット",
    "bolt": "ボルト",
    "spacer": "スペーサー",
    "washer": "ワッシャ",
}
PART_ORDER = tuple(PART_LABELS)
FRAME_WIDTH = 800
FRAME_HEIGHT = 600


@dataclass(frozen=True)
class AnimationSpeed:
    step_seconds: float
    decision_seconds: float


SPEED_PRESETS = {
    "ステップ表示": AnimationSpeed(0.10, 0.30),
    "結果のみ": AnimationSpeed(0.0, 0.0),
}


@dataclass(frozen=True)
class DisplayDetection:
    source_index: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    ordinal: int

    @property
    def label(self) -> str:
        return f"{PART_LABELS.get(self.class_name, self.class_name)}{self.ordinal}"


def ordered_detections(model_result: Any) -> tuple[DisplayDetection, ...]:
    """クラス→中心x→中心y→元インデックスの順。元の結果を並べ替えない。"""
    ranks = {name: rank for rank, name in enumerate(PART_ORDER)}
    raw = []
    for index, box in enumerate(model_result.boxes):
        name = model_result.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
        if len(coordinates) != 4 or not all(isfinite(value) for value in coordinates):
            raise ValueError(f"検出枠の座標が不正です: index={index}")
        x1, y1, x2, y2 = coordinates
        if x2 < x1 or y2 < y1 or not isfinite(confidence):
            raise ValueError(f"検出結果が不正です: index={index}")
        raw.append((index, name, confidence, coordinates))
    raw.sort(key=lambda item: (
        ranks.get(item[1], len(ranks)),
        item[1] if item[1] not in ranks else "",
        (item[3][0] + item[3][2]) / 2,
        (item[3][1] + item[3][3]) / 2,
        item[0],
    ))
    ordinals: dict[str, int] = {}
    ordered = []
    for index, name, confidence, coordinates in raw:
        ordinals[name] = ordinals.get(name, 0) + 1
        ordered.append(DisplayDetection(index, name, confidence, coordinates, ordinals[name]))
    return tuple(ordered)


def load_original_image(image_path: str | Path) -> np.ndarray:
    """日本語を含むパスから表示用に読む。推論用の入力は変更しない。"""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    return image


def render_detection_frame(
    original_bgr: np.ndarray,
    visible: Sequence[DisplayDetection] = (),
    *,
    highlight_last: bool = False,
) -> np.ndarray:
    """固定サイズのBGR表示画像を1枚作る。座標にも同じ拡縮・余白を適用する。"""
    height, width = original_bgr.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("空の画像は表示できません。")
    scale = min(FRAME_WIDTH / width, FRAME_HEIGHT / height, 1.0)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (FRAME_WIDTH - resized_width) // 2
    top = (FRAME_HEIGHT - resized_height) // 2
    canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (32, 17, 11), dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        original_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA,
    )
    # 丸め後の実際の横・縦倍率を使い、元のbboxデータは変更しない。
    scale_x, scale_y = resized_width / width, resized_height / height
    labels = []
    for position, detection in enumerate(visible):
        x1, y1, x2, y2 = detection.xyxy
        start = (left + round(x1 * scale_x), top + round(y1 * scale_y))
        end = (left + round(x2 * scale_x), top + round(y2 * scale_y))
        newest = highlight_last and position == len(visible) - 1
        color = (36, 191, 251) if newest else (248, 189, 56)
        cv2.rectangle(canvas, start, end, color, 4 if newest else 2)
        labels.append((start, detection))
    # 後から描く枠がラベル文字を横切らないよう、ラベルは全枠の後に描く。
    occupied = []
    for start, detection in labels:
        text = f"{detection.class_name}{detection.ordinal} {detection.confidence:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
        text_x = max(4, min(start[0], FRAME_WIDTH - text_width - 5))
        text_y = max(text_height + 5, min(start[1] - 7, FRAME_HEIGHT - baseline - 5))
        line_height = text_height + baseline + 10
        for offset in (0, *[sign * row * line_height for row in range(1, 20) for sign in (-1, 1)]):
            candidate_y = text_y + offset
            if not text_height + 5 <= candidate_y <= FRAME_HEIGHT - baseline - 5:
                continue
            candidate = (text_x - 4, candidate_y - text_height - 4,
                         text_x + text_width + 4, candidate_y + baseline + 4)
            if not any(candidate[0] < box[2] and candidate[2] > box[0]
                       and candidate[1] < box[3] and candidate[3] > box[1] for box in occupied):
                text_y = candidate_y
                break
        occupied.append((text_x - 4, text_y - text_height - 4,
                         text_x + text_width + 4, text_y + baseline + 4))
        cv2.rectangle(canvas, (text_x - 4, text_y - text_height - 4),
                      (text_x + text_width + 4, text_y + baseline + 4), (35, 35, 35), -1)
        cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def play_detection_steps(
    model_result: Any,
    speed: AnimationSpeed,
    display: Callable[[np.ndarray, tuple[DisplayDetection, ...], int], None],
    *,
    wait: Callable[[float], None] = sleep,
) -> tuple[DisplayDetection, ...]:
    """その画像のフレームを逐次生成する。演出なし・検出0件では1枚だけ表示する。"""
    detections = ordered_detections(model_result)
    steps = range(1, len(detections) + 1) if speed.step_seconds > 0 and detections else [len(detections)]
    for count in steps:
        visible = detections[:count]
        frame = render_detection_frame(
            model_result.orig_img, visible, highlight_last=speed.step_seconds > 0,
        )
        display(frame, visible, len(detections))
        if speed.step_seconds > 0 and visible:
            wait(speed.step_seconds)
    return detections


def hold_decision(speed: AnimationSpeed, *, wait: Callable[[float], None] = sleep) -> None:
    if speed.decision_seconds > 0:
        wait(speed.decision_seconds)


def quantity_comparison(counts: dict[str, int]) -> list[dict[str, str | int]]:
    """全検出枠の表示後にだけ使う、既存判定数量の説明。再判定はしない。"""
    rows = []
    for name in PART_ORDER:
        actual = counts[name]
        expected = EXPECTED_COUNTS[name]
        status = "一致" if actual == expected else ("未検出" if actual == 0 else "数量相違")
        rows.append({"部品": PART_LABELS[name], "検出数": actual, "期待数": expected, "照合": status})
    return rows
