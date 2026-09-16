"""検出枠の幾何からワッシャの側を求める。画像AI・DB・UIには依存しない。"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isclose, isfinite
from typing import Protocol


class PositionedDetection(Protocol):
    class_name: str
    xyxy: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class AssemblyAxis:
    origin: tuple[float, float]
    direction: tuple[float, float]
    length: float

    def project(self, point: tuple[float, float]) -> float:
        """ボルト中心を0、化粧ナット中心を1とする軸上の位置。"""
        dx, dy = point[0] - self.origin[0], point[1] - self.origin[1]
        return (dx * self.direction[0] + dy * self.direction[1]) / self.length


@dataclass(frozen=True)
class WasherProjection:
    detection_index: int
    center: float
    interval: tuple[float, float]
    side: str


@dataclass(frozen=True)
class WasherPosition:
    status: str
    reason: str
    axis: AssemblyAxis | None = None
    spacer_center: float | None = None
    washers: tuple[WasherProjection, ...] = ()
    missing_defect_type: str | None = None


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 / 2 + x2 / 2, y1 / 2 + y2 / 2)


def _corners(box: tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return ((x1, y1), (x1, y2), (x2, y1), (x2, y2))


def _interval(axis: AssemblyAxis, box: tuple[float, float, float, float]) -> tuple[float, float]:
    values = [axis.project(point) for point in _corners(box)]
    return min(values), max(values)


def _crosses_axis(axis: AssemblyAxis, box: tuple[float, float, float, float]) -> bool:
    offsets = [
        (x - axis.origin[0]) * -axis.direction[1]
        + (y - axis.origin[1]) * axis.direction[0]
        for x, y in _corners(box)
    ]
    return min(offsets) <= 0 <= max(offsets)


def _axis_intersection(axis: AssemblyAxis, box: tuple[float, float, float, float]) -> tuple[float, float]:
    """組付け軸が枠を横切る区間。斜め枠の背景の角は含めない。

    呼び出し元は中心が軸上にある基準部品に限定する。
    """
    lower, upper = float('-inf'), float('inf')
    for origin, direction, minimum, maximum in zip(axis.origin, axis.direction, box[:2], box[2:]):
        if direction == 0:
            continue
        first, last = sorted(((minimum - origin) / (direction * axis.length),
                              (maximum - origin) / (direction * axis.length)))
        lower, upper = max(lower, first), min(upper, last)
    return lower, upper


def analyze_washer_position(detections: Iterable[PositionedDetection]) -> WasherPosition:
    """明確な1枚欠品だけ側を特定する。根拠不足なら既存数量判定へ委ねる。

    軸はボルト中心→化粧ナット中心。ワッシャとスペーサーの中心の順序で
    分類し、中心の一致・軸外・軸上の基準枠の重なりは保留する。
    水平垂直の検出枠は斜め部品の背景を含むため、その四隅の射影を部品の
    実際の長さや位置の誤差範囲とは扱わない。intervalは診断用に残す。
    confidenceを幾何の確信度へ変換したり、経験的な画素しきい値を置いたりしない。
    """
    groups: dict[str, list[tuple[int, tuple | None]]] = {name: [] for name in ("bolt", "cap_nut", "spacer", "washer")}
    for index, detection in enumerate(detections):
        if detection.class_name in groups:
            groups[detection.class_name].append((index, detection.xyxy))
    if any(len(groups[name]) != 1 for name in ("bolt", "cap_nut", "spacer")):
        return WasherPosition("unresolved", "anchor_count")
    if len(groups["washer"]) not in (1, 2):
        return WasherPosition("unresolved", "washer_count")
    for group in groups.values():
        for _, box in group:
            if box is None:
                return WasherPosition("unresolved", "coordinates_unavailable")
            if len(box) != 4 or not all(isfinite(value) for value in box):
                return WasherPosition("unresolved", "invalid_coordinates")
            if box[0] >= box[2] or box[1] >= box[3]:
                return WasherPosition("unresolved", "invalid_coordinates")

    bolt, cap, spacer = (groups[name][0][1] for name in ("bolt", "cap_nut", "spacer"))
    origin, end = _center(bolt), _center(cap)
    dx, dy = end[0] - origin[0], end[1] - origin[1]
    length = hypot(dx, dy)
    if length == 0 or not isfinite(length):
        return WasherPosition("unresolved", "axis_unavailable")
    axis = AssemblyAxis(origin, (dx / length, dy / length), length)
    if _axis_intersection(axis, bolt)[1] >= _axis_intersection(axis, cap)[0]:
        return WasherPosition("unresolved", "anchors_overlap", axis)
    spacer_center = axis.project(_center(spacer))
    if not 0 < spacer_center < 1 or not _crosses_axis(axis, spacer):
        return WasherPosition("unresolved", "spacer_outside_axis", axis, spacer_center)

    projections = []
    for index, box in groups["washer"]:
        center = axis.project(_center(box))
        lower, upper = _interval(axis, box)
        if not 0 < center < 1 or not _crosses_axis(axis, box):
            side = "outside_axis"
        elif isclose(center, spacer_center, rel_tol=0, abs_tol=1e-9):
            side = "ambiguous"
        elif center < spacer_center:
            side = "before_spacer"
        elif center > spacer_center:
            side = "after_spacer"
        else:
            side = "ambiguous"
        projections.append(WasherProjection(index, center, (lower, upper), side))
    washers = tuple(sorted(projections, key=lambda item: (item.center, item.detection_index)))
    if any(item.side in ("ambiguous", "outside_axis") for item in washers):
        return WasherPosition("unresolved", "washer_position_ambiguous", axis, spacer_center, washers)
    if len(washers) == 2:
        if {item.side for item in washers} != {"before_spacer", "after_spacer"}:
            return WasherPosition("unresolved", "washers_on_same_side", axis, spacer_center, washers)
        return WasherPosition("resolved", "both_sides_present", axis, spacer_center, washers)
    missing = (
        "washer_after_spacer_missing" if washers[0].side == "before_spacer"
        else "washer_before_spacer_missing"
    )
    return WasherPosition("resolved", "single_side_missing", axis, spacer_center, washers, missing)
