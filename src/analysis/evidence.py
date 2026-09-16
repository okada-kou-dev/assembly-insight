from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from src.analysis.quality import (
    calculate_defect_occurrence_summary,
    calculate_quality_trend_patterns,
    calculate_time_band_quality_metrics,
)

InspectionRow = Mapping[str, Any]


def _percent(rate: float) -> float:
    return round(rate * 100.0, 1)


def _percentage_point_change(
    after_rate: float,
    before_rate: float,
) -> float:
    return round(
        (after_rate - before_rate) * 100.0,
        1,
    )


def _build_time_band_concentration(
    time_band_metrics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    defect_types: set[str] = set()

    for metrics in time_band_metrics:
        defect_types.update(metrics["defect_counts"].keys())

    best_candidate: dict[str, Any] | None = None
    best_difference = 0.0

    for defect_type in sorted(defect_types):
        for metrics in time_band_metrics:
            band_total = metrics["total_inspections"]

            if band_total == 0:
                continue

            band_count = metrics["defect_counts"].get(defect_type, 0)

            band_rate = band_count / band_total

            other_total = sum(
                row["total_inspections"]
                for row in time_band_metrics
                if row["time_band"] != metrics["time_band"]
            )

            other_count = sum(
                row["defect_counts"].get(
                    defect_type,
                    0,
                )
                for row in time_band_metrics
                if row["time_band"] != metrics["time_band"]
            )

            if other_total == 0:
                continue

            other_rate = other_count / other_total

            difference = band_rate - other_rate

            if difference > best_difference:
                best_difference = difference

                best_candidate = {
                    "type": ("time_band_defect_concentration"),
                    "summary": (
                        f"{metrics['time_band']}で"
                        f"{defect_type}が"
                        "他時間帯より集中している傾向"
                    ),
                    "facts": {
                        "time_band": metrics["time_band"],
                        "defect_type": defect_type,
                        "band_count": band_count,
                        "band_total": band_total,
                        "band_rate_percent": (_percent(band_rate)),
                        "other_count": other_count,
                        "other_total": other_total,
                        "other_rate_percent": (_percent(other_rate)),
                        ("rate_difference_percentage_points"): round(
                            difference * 100.0,
                            1,
                        ),
                    },
                }

    return best_candidate


def build_quality_evidence(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    Python品質分析結果からLLM参照用evidenceを生成する。

    数値・割合・日付・比較値はすべてPython側で確定する。
    """
    inspection_list = list(inspections)

    if not inspection_list:
        return []

    evidence_candidates: list[dict[str, Any]] = []

    time_band_metrics = calculate_time_band_quality_metrics(inspection_list)

    concentration = _build_time_band_concentration(time_band_metrics)

    if concentration is not None:
        evidence_candidates.append(concentration)

    occurrence_summaries = calculate_defect_occurrence_summary(inspection_list)

    for occurrence in occurrence_summaries:
        if occurrence["present_on_latest_date"]:
            continue

        evidence_candidates.append(
            {
                "type": "defect_disappearance",
                # 最終発生日を未発生期間に含めず、検査履歴の範囲に限定する。
                "summary": (
                    "提供された検査履歴では、"
                    f"{occurrence['defect_type']}は"
                    f"{occurrence['last_date']}を最後に、"
                    "その後の観測期間では記録されていない"
                ),
                "facts": {
                    "defect_type": occurrence["defect_type"],
                    "total_count": occurrence["total_count"],
                    "first_date": occurrence["first_date"],
                    "last_date": occurrence["last_date"],
                    "days_since_last_occurrence": (
                        occurrence["days_since_last_occurrence"]
                    ),
                },
            }
        )

    trend_patterns = calculate_quality_trend_patterns(inspection_list)

    post_peak = trend_patterns["post_peak"]

    if post_peak is not None and post_peak["recovered"]:
        evidence_candidates.append(
            {
                "type": "peak_recovery",
                "summary": ("不良率はピーク後に低下している"),
                "facts": {
                    "peak_date": post_peak["peak_date"],
                    "peak_defect_rate_percent": (
                        _percent(post_peak["peak_defect_rate"])
                    ),
                    "latest_date": post_peak["latest_date"],
                    "latest_defect_rate_percent": (
                        _percent(post_peak["latest_defect_rate"])
                    ),
                    ("defect_rate_change_percentage_points"): _percentage_point_change(
                        post_peak["latest_defect_rate"],
                        post_peak["peak_defect_rate"],
                    ),
                },
            }
        )

    increase = trend_patterns["longest_increase_streak"]

    if increase is not None:
        evidence_candidates.append(
            {
                "type": ("consecutive_defect_rate_increase"),
                "summary": (
                    "検査記録のある日を日付順に並べた隣接比較で、"
                    f"日別不良率が{increase['consecutive_increases']}回連続で"
                    "上昇している期間がある"
                ),
                "facts": {
                    "start_date": increase["start_date"],
                    "end_date": increase["end_date"],
                    "consecutive_increases": (increase["consecutive_increases"]),
                    "start_defect_rate_percent": (
                        _percent(increase["start_defect_rate"])
                    ),
                    "end_defect_rate_percent": (_percent(increase["end_defect_rate"])),
                    ("defect_rate_change_percentage_points"): _percentage_point_change(
                        increase["end_defect_rate"],
                        increase["start_defect_rate"],
                    ),
                },
            }
        )

    evidence: list[dict[str, Any]] = []

    for index, candidate in enumerate(
        evidence_candidates,
        start=1,
    ):
        evidence.append(
            {
                "evidence_id": f"F{index:02d}",
                **candidate,
            }
        )

    return evidence
