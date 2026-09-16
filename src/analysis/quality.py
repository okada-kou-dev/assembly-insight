from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from src.storage.database import list_inspections

InspectionRow = Mapping[str, Any]


def calculate_quality_metrics(
    inspections: Iterable[InspectionRow],
) -> dict[str, Any]:
    """
    検査履歴から基本品質指標を決定論的に計算する。

    defect_rate は 0.0 ～ 1.0 の比率で返す。
    defect_counts は NG レコードの defect_type のみ集計する。
    """
    total_inspections = 0
    ok_count = 0
    ng_count = 0
    defect_counter: Counter[str] = Counter()

    for inspection in inspections:
        total_inspections += 1

        inspection_result = inspection["inspection_result"]

        if inspection_result == "OK":
            ok_count += 1

        elif inspection_result == "NG":
            ng_count += 1

            defect_type = inspection["defect_type"]
            if defect_type:
                defect_counter[str(defect_type)] += 1

        else:
            raise ValueError(f"Unexpected inspection_result: {inspection_result!r}")

    if total_inspections == 0:
        defect_rate = 0.0
    else:
        defect_rate = ng_count / total_inspections

    return {
        "total_inspections": total_inspections,
        "ok_count": ok_count,
        "ng_count": ng_count,
        "defect_rate": defect_rate,
        "defect_counts": dict(sorted(defect_counter.items())),
    }


def load_quality_metrics(
    db_path: str | Path,
) -> dict[str, Any]:
    """
    SQLiteに保存された検査履歴を取得し、
    基本品質指標を計算する。
    """
    inspections = list_inspections(db_path)
    return calculate_quality_metrics(inspections)


def calculate_daily_quality_metrics(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    検査履歴を captured_at の日付単位で集計する。

    戻り値は日付昇順。
    defect_rate は 0.0 ～ 1.0 の比率で返す。
    """
    inspections_by_date: dict[str, list[InspectionRow]] = {}

    for inspection in inspections:
        captured_at = inspection["captured_at"]

        try:
            captured_date = datetime.fromisoformat(str(captured_at)).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at: {captured_at!r}") from exc

        inspections_by_date.setdefault(
            captured_date,
            [],
        ).append(inspection)

    daily_metrics: list[dict[str, Any]] = []

    for captured_date in sorted(inspections_by_date):
        metrics = calculate_quality_metrics(inspections_by_date[captured_date])

        daily_metrics.append(
            {
                "date": captured_date,
                "total_inspections": metrics["total_inspections"],
                "ok_count": metrics["ok_count"],
                "ng_count": metrics["ng_count"],
                "defect_rate": metrics["defect_rate"],
            }
        )

    return daily_metrics


def load_daily_quality_metrics(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    SQLiteの検査履歴を取得し、
    日別品質指標を計算する。
    """
    inspections = list_inspections(db_path)
    return calculate_daily_quality_metrics(inspections)


TIME_BAND_LABELS = {
    "00_06": "00:00–06:00",
    "06_12": "06:00–12:00",
    "12_18": "12:00–18:00",
    "18_24": "18:00–24:00",
}
TIME_BAND_ORDER = tuple(TIME_BAND_LABELS)


def classify_time_band(hour: int) -> str:
    """暦日の24時間を6時間ずつに等分する。境界時刻は次の区間へ入る。"""
    if not isinstance(hour, int) or not 0 <= hour < 24:
        raise ValueError(f"Invalid hour: {hour!r}")
    return TIME_BAND_ORDER[hour // 6]


def calculate_time_band_quality_metrics(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    検査履歴を時間帯別に集計する。

    各時間帯について、
    total_inspections / ok_count / ng_count /
    defect_rate / defect_counts をPythonで決定論的に計算する。

    データが存在しない時間帯も0件として返す。
    """
    inspections_by_time_band: dict[str, list[InspectionRow]] = {
        time_band: [] for time_band in TIME_BAND_ORDER
    }

    for inspection in inspections:
        captured_at = inspection["captured_at"]

        try:
            captured_datetime = datetime.fromisoformat(str(captured_at))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at: {captured_at!r}") from exc

        time_band = classify_time_band(captured_datetime.hour)

        inspections_by_time_band[time_band].append(inspection)

    time_band_metrics: list[dict[str, Any]] = []

    for time_band in TIME_BAND_ORDER:
        metrics = calculate_quality_metrics(inspections_by_time_band[time_band])

        time_band_metrics.append(
            {
                "time_band": time_band,
                "total_inspections": metrics["total_inspections"],
                "ok_count": metrics["ok_count"],
                "ng_count": metrics["ng_count"],
                "defect_rate": metrics["defect_rate"],
                "defect_counts": metrics["defect_counts"],
            }
        )

    return time_band_metrics


def load_time_band_quality_metrics(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    SQLiteの検査履歴を取得し、
    時間帯別品質指標を計算する。
    """
    inspections = list_inspections(db_path)

    return calculate_time_band_quality_metrics(inspections)


def build_quality_trend_data(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    検査履歴から時系列グラフ用の日別データを生成する。

    日付昇順で返し、不良率は0.0～1.0の値に加えて
    表示用の百分率もPython側で計算する。
    """
    daily_metrics = calculate_daily_quality_metrics(inspections)

    trend_data: list[dict[str, Any]] = []

    for metrics in daily_metrics:
        trend_data.append(
            {
                "date": metrics["date"],
                "total_inspections": metrics["total_inspections"],
                "ok_count": metrics["ok_count"],
                "ng_count": metrics["ng_count"],
                "defect_rate": metrics["defect_rate"],
                "defect_rate_percent": (metrics["defect_rate"] * 100.0),
            }
        )

    return trend_data


def load_quality_trend_data(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    SQLite履歴から時系列グラフ用データを生成する。
    """
    inspections = list_inspections(db_path)

    return build_quality_trend_data(inspections)


def calculate_defect_occurrence_summary(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    不良種類ごとの発生期間を決定論的に集計する。

    各不良について以下を返す。
    - total_count
    - first_date
    - last_date
    - present_on_latest_date
    - days_since_last_occurrence
    """
    parsed_inspections: list[tuple[InspectionRow, datetime]] = []

    for inspection in inspections:
        captured_at = inspection["captured_at"]

        try:
            captured_datetime = datetime.fromisoformat(str(captured_at))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at: {captured_at!r}") from exc

        parsed_inspections.append((inspection, captured_datetime))

    if not parsed_inspections:
        return []

    latest_inspection_date = max(
        captured_datetime.date() for _, captured_datetime in parsed_inspections
    )

    defect_dates: dict[str, list] = {}

    for inspection, captured_datetime in parsed_inspections:
        if inspection["inspection_result"] != "NG":
            continue

        defect_type = inspection["defect_type"]

        if not defect_type:
            continue

        defect_dates.setdefault(
            str(defect_type),
            [],
        ).append(captured_datetime.date())

    summaries: list[dict[str, Any]] = []

    for defect_type in sorted(defect_dates):
        dates = defect_dates[defect_type]

        first_date = min(dates)
        last_date = max(dates)

        summaries.append(
            {
                "defect_type": defect_type,
                "total_count": len(dates),
                "first_date": first_date.isoformat(),
                "last_date": last_date.isoformat(),
                "present_on_latest_date": (last_date == latest_inspection_date),
                "days_since_last_occurrence": (latest_inspection_date - last_date).days,
            }
        )

    return summaries


def load_defect_occurrence_summary(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    SQLite履歴から不良種類別の発生期間を集計する。
    """
    inspections = list_inspections(db_path)

    return calculate_defect_occurrence_summary(inspections)


def calculate_daily_defect_counts(
    inspections: Iterable[InspectionRow],
) -> list[dict[str, Any]]:
    """
    検査履歴から日別の不良種類別件数を集計する。

    日付昇順で返す。
    不良件数は既存の calculate_quality_metrics() を再利用して計算する。
    """
    inspections_by_date: dict[str, list[InspectionRow]] = {}

    for inspection in inspections:
        captured_at = inspection["captured_at"]

        try:
            captured_date = datetime.fromisoformat(str(captured_at)).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at: {captured_at!r}") from exc

        inspections_by_date.setdefault(
            captured_date,
            [],
        ).append(inspection)

    daily_defect_counts: list[dict[str, Any]] = []

    for captured_date in sorted(inspections_by_date):
        metrics = calculate_quality_metrics(inspections_by_date[captured_date])

        daily_defect_counts.append(
            {
                "date": captured_date,
                "total_inspections": metrics["total_inspections"],
                "ng_count": metrics["ng_count"],
                "defect_counts": metrics["defect_counts"],
            }
        )

    return daily_defect_counts


def load_daily_defect_counts(
    db_path: str | Path,
) -> list[dict[str, Any]]:
    """
    SQLite履歴から日別の不良種類別件数を取得する。
    """
    inspections = list_inspections(db_path)

    return calculate_daily_defect_counts(inspections)


def calculate_quality_trend_patterns(
    inspections: Iterable[InspectionRow],
) -> dict[str, Any]:
    """
    日別不良率からピーク、ピーク後変化、
    最長の連続上昇・連続低下を抽出する。

    数値計算はすべてPython側で行う。
    """
    daily_metrics = calculate_daily_quality_metrics(inspections)

    if not daily_metrics:
        return {
            "peak": None,
            "post_peak": None,
            "longest_increase_streak": None,
            "longest_decrease_streak": None,
        }

    peak_metrics = max(
        daily_metrics,
        key=lambda row: row["defect_rate"],
    )

    peak_index = daily_metrics.index(peak_metrics)

    post_peak = None

    if peak_index < len(daily_metrics) - 1:
        latest_metrics = daily_metrics[-1]

        rate_change = latest_metrics["defect_rate"] - peak_metrics["defect_rate"]

        post_peak = {
            "peak_date": peak_metrics["date"],
            "peak_defect_rate": peak_metrics["defect_rate"],
            "latest_date": latest_metrics["date"],
            "latest_defect_rate": latest_metrics["defect_rate"],
            "defect_rate_change": rate_change,
            "recovered": rate_change < 0.0,
        }

    longest_increase_streak = None
    longest_decrease_streak = None

    increase_start_index = 0
    increase_steps = 0
    best_increase_steps = 0
    best_increase_start_index = 0
    best_increase_end_index = 0

    decrease_start_index = 0
    decrease_steps = 0
    best_decrease_steps = 0
    best_decrease_start_index = 0
    best_decrease_end_index = 0

    for index in range(1, len(daily_metrics)):
        previous_rate = daily_metrics[index - 1]["defect_rate"]

        current_rate = daily_metrics[index]["defect_rate"]

        if current_rate > previous_rate:
            if increase_steps == 0:
                increase_start_index = index - 1

            increase_steps += 1

            if increase_steps > best_increase_steps:
                best_increase_steps = increase_steps
                best_increase_start_index = increase_start_index
                best_increase_end_index = index
        else:
            increase_steps = 0

        if current_rate < previous_rate:
            if decrease_steps == 0:
                decrease_start_index = index - 1

            decrease_steps += 1

            if decrease_steps > best_decrease_steps:
                best_decrease_steps = decrease_steps
                best_decrease_start_index = decrease_start_index
                best_decrease_end_index = index
        else:
            decrease_steps = 0

    if best_increase_steps > 0:
        longest_increase_streak = {
            "start_date": daily_metrics[best_increase_start_index]["date"],
            "end_date": daily_metrics[best_increase_end_index]["date"],
            "consecutive_increases": (best_increase_steps),
            "start_defect_rate": daily_metrics[best_increase_start_index][
                "defect_rate"
            ],
            "end_defect_rate": daily_metrics[best_increase_end_index]["defect_rate"],
        }

    if best_decrease_steps > 0:
        longest_decrease_streak = {
            "start_date": daily_metrics[best_decrease_start_index]["date"],
            "end_date": daily_metrics[best_decrease_end_index]["date"],
            "consecutive_decreases": (best_decrease_steps),
            "start_defect_rate": daily_metrics[best_decrease_start_index][
                "defect_rate"
            ],
            "end_defect_rate": daily_metrics[best_decrease_end_index]["defect_rate"],
        }

    return {
        "peak": {
            "date": peak_metrics["date"],
            "defect_rate": peak_metrics["defect_rate"],
            "ng_count": peak_metrics["ng_count"],
            "total_inspections": peak_metrics["total_inspections"],
        },
        "post_peak": post_peak,
        "longest_increase_streak": (longest_increase_streak),
        "longest_decrease_streak": (longest_decrease_streak),
    }


def load_quality_trend_patterns(
    db_path: str | Path,
) -> dict[str, Any]:
    """
    SQLite履歴から品質時系列パターンを抽出する。
    """
    inspections = list_inspections(db_path)

    return calculate_quality_trend_patterns(inspections)
