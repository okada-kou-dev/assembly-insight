"""品質の推定幅とNG構成の偏りを、Pythonだけで計算する。"""

from math import sqrt
from statistics import NormalDist

from src.analysis.quality import TIME_BAND_LABELS

CONFIDENCE_LEVEL = 0.95
WILSON_Z = NormalDist().inv_cdf((1 + CONFIDENCE_LEVEL) / 2)


def wilson_interval(ng_count: int, total: int) -> tuple[float, float] | None:
    """二項比率の両側95% Wilson区間。分母0は推定しない。"""
    if not isinstance(ng_count, int) or not isinstance(total, int) or not 0 <= ng_count <= total:
        raise ValueError("NG件数と検査件数は0以上の整数で、NG件数は検査件数以下である必要があります。")
    if total == 0:
        return None
    rate = ng_count / total
    correction = WILSON_Z ** 2 / total
    center = (rate + correction / 2) / (1 + correction)
    half_width = WILSON_Z * sqrt(rate * (1 - rate) / total + WILSON_Z ** 2 / (4 * total ** 2)) / (1 + correction)
    return max(0.0, center - half_width), min(1.0, center + half_width)


def confidence_rows(bands: list[dict]) -> list[dict]:
    rows = []
    for band in bands:
        total, ng = band["total_inspections"], band["ng_count"]
        interval = wilson_interval(ng, total)
        rows.append({"band": TIME_BAND_LABELS[band["time_band"]], "total": total, "ng": ng,
                     "rate": ng / total if total else None,
                     "lower": interval[0] if interval else None,
                     "upper": interval[1] if interval else None})
    return rows


def ng_category_counts(metrics: dict) -> dict[str, int]:
    """未分類NGを補い、NG総数と分類合計を一致させる。元の辞書は変更しない。"""
    counts = {key: count for key, count in metrics["defect_counts"].items() if count}
    missing = metrics["ng_count"] - sum(counts.values())
    if missing < 0 or any(count < 0 for count in counts.values()):
        raise ValueError("NG件数と不良分類の件数が一致しません。")
    if missing:
        counts["unknown"] = counts.get("unknown", 0) + missing
    return counts


def residual_analysis(bands: list[dict]) -> dict:
    """NGだけの時間帯×分類表。E=行合計×列合計/N、Pearson残差=(O-E)/sqrt(E)。"""
    counts_by_band = [ng_category_counts(band) for band in bands]
    keys = sorted({key for counts in counts_by_band for key in counts})
    ng_total = sum(band["ng_count"] for band in bands)
    active_bands = sum(band["ng_count"] > 0 for band in bands)
    degrees = max(0, active_bands - 1) * max(0, len(keys) - 1)
    if not ng_total or not degrees:
        return {"available": False, "ng_total": ng_total, "degrees_of_freedom": degrees,
                "chi_square": None, "small_expected_cells": 0, "cells": []}
    column_totals = {key: sum(counts.get(key, 0) for counts in counts_by_band) for key in keys}
    cells = []
    for band, counts in zip(bands, counts_by_band):
        if band["ng_count"] == 0:
            continue
        for key in keys:
            expected = band["ng_count"] * column_totals[key] / ng_total
            observed = counts.get(key, 0)
            residual = (observed - expected) / sqrt(expected)
            cells.append({"band": TIME_BAND_LABELS[band["time_band"]], "key": key,
                          "observed": observed, "expected": expected, "residual": residual,
                          "contribution": residual ** 2, "band_ng": band["ng_count"],
                          "category_total": column_totals[key], "ng_total": ng_total})
    return {"available": True, "ng_total": ng_total, "degrees_of_freedom": degrees,
            "chi_square": sum(cell["contribution"] for cell in cells),
            "small_expected_cells": sum(cell["expected"] < 5 for cell in cells), "cells": cells}
