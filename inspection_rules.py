from collections.abc import Mapping
from dataclasses import dataclass

EXPECTED_COUNTS = {
    "bolt": 1,
    "washer": 2,
    "spacer": 1,
    "cap_nut": 1,
}


MISSING_DEFECT_TYPES = {
    "bolt": "bolt_missing",
    "washer": "washer_missing",
    "spacer": "spacer_missing",
    "cap_nut": "cap_nut_missing",
}


@dataclass(frozen=True)
class InspectionDecision:
    inspection_result: str
    defect_type: str | None
    counts: dict[str, int]


def judge_part_counts(counts: Mapping[str, int]) -> InspectionDecision:
    """
    AIが検出した部品数量から組付けOK / NGを判定する。

    Parameters
    ----------
    counts:
        {
            "bolt": 1,
            "washer": 2,
            "spacer": 1,
            "cap_nut": 1,
        }

    Returns
    -------
    InspectionDecision
        inspection_result: "OK" または "NG"
        defect_type: 不良種類。正常時は None
        counts: 判定に使用した正規化後の数量
    """

    # 検出されなかったクラスは0個として扱う
    normalized_counts = {part: counts.get(part, 0) for part in EXPECTED_COUNTS}

    # 不正な数量が渡された場合は早めにエラーにする
    for part, count in normalized_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Count for {part!r} must be a non-negative integer: {count!r}"
            )

    # 正常数量と完全一致
    if normalized_counts == EXPECTED_COUNTS:
        return InspectionDecision(
            inspection_result="OK",
            defect_type=None,
            counts=normalized_counts,
        )

    missing_parts = [
        part
        for part, expected in EXPECTED_COUNTS.items()
        if normalized_counts[part] < expected
    ]

    excess_parts = [
        part
        for part, expected in EXPECTED_COUNTS.items()
        if normalized_counts[part] > expected
    ]

    # 数量過多は今回のMVP判定対象ではないが、
    # 正常数量とは一致しないため安全側でNGにする
    if excess_parts:
        defect_type = "quantity_mismatch"

    # 欠品が1種類だけなら具体的な defect_type
    elif len(missing_parts) == 1:
        defect_type = MISSING_DEFECT_TYPES[missing_parts[0]]

    # 複数種類の欠品
    else:
        defect_type = "multiple_missing"

    return InspectionDecision(
        inspection_result="NG",
        defect_type=defect_type,
        counts=normalized_counts,
    )
