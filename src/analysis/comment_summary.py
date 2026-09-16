"""自由コメント用に、検査履歴を分母付きのPython集計表へ整形する。"""

import csv
import io
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from src.analysis.quality import (
    InspectionRow,
    TIME_BAND_LABELS,
    calculate_quality_metrics,
    calculate_time_band_quality_metrics,
)
from src.analysis.statistics import confidence_rows, ng_category_counts, residual_analysis

DEFECT_LABELS = {
    "washer_missing": "ワッシャ欠品（位置未確定）",
    "washer_before_spacer_missing": "ワッシャ欠品（ボルト側）",
    "washer_after_spacer_missing": "ワッシャ欠品（化粧ナット側）",
    "spacer_missing": "スペーサー欠品",
    "cap_nut_missing": "化粧ナット欠品",
    "bolt_missing": "ボルト欠品",
    "multiple_missing": "複数部品欠品",
    "quantity_mismatch": "数量異常",
    "unknown": "不良種別未記録",
}



def _ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "算出不可(分母0件)"
    return f"{numerator}/{denominator}={100 * numerator / denominator:.1f}%"


def _csv_text(rows: list[list[Any]]) -> str:
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def _dashboard_summary(overall: dict, bands: list[dict]) -> str:
    """現行の構成比・推定幅・残差図に対応する計算済みの表。"""
    ranking = [["表示順", "不良種別", "件数", "全NG件数", "NG内構成比", "上位からの累積構成比"]]
    cumulative = 0
    counts = ng_category_counts(overall)
    for order, (key, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1):
        cumulative += count
        ranking.append([order, DEFECT_LABELS.get(key, key), count, overall['ng_count'],
                        _ratio(count, overall['ng_count']), _ratio(cumulative, overall['ng_count'])])

    intervals = [["時間帯", "検査件数", "NG件数", "不良率(NG/検査)", "95%区間下限", "95%区間上限"]]
    for row in confidence_rows(bands):
        intervals.append([row['band'], row['total'], row['ng'], _ratio(row['ng'], row['total']),
                          f"{row['lower']:.1%}" if row['lower'] is not None else "算出不可(分母0件)",
                          f"{row['upper']:.1%}" if row['upper'] is not None else "算出不可(分母0件)"])

    residuals = residual_analysis(bands)
    residual_table = [["時間帯", "不良種別", "実件数", "期待件数", "Pearson残差", "χ²寄与"]]
    for cell in residuals['cells']:
        residual_table.append([cell['band'], DEFECT_LABELS.get(cell['key'], cell['key']),
                               cell['observed'], f"{cell['expected']:.2f}", f"{cell['residual']:+.2f}",
                               f"{cell['contribution']:.2f}"])
    residual_totals = [["対象NG件数", "Pearsonχ²", "自由度", "期待件数5未満のセル数", "全セル数"],
                       [residuals['ng_total'], f"{residuals['chi_square']:.2f}" if residuals['available'] else "算出不可",
                        residuals['degrees_of_freedom'], residuals['small_expected_cells'], len(residuals['cells'])]]
    return (
        "\n不良内訳・累積構成比(CSV)\n" + _csv_text(ranking)
        + "\n時間帯別不良率の95% Wilson信頼区間(CSV)\n" + _csv_text(intervals)
        + "\n不良構成の残差分析・全体(CSV)\n" + _csv_text(residual_totals)
        + "\n不良構成の残差分析・時間帯×種別(CSV)\n" + _csv_text(residual_table)
        + "\n統計図の定義：95% Wilson区間は各時間帯のNG/検査件数の推定幅。"
        "残差分析はNG内の構成で、期待件数は時間帯と不良種別の独立を仮定した値。"
        "Pearson残差は(実件数−期待件数)/√期待件数、χ²寄与は残差の二乗。"
        "いずれも独立した検査を前提とする参考値で、p値・有意差判定・原因の特定は含まない。\n"
    )


def build_quality_summary(
    inspections: Iterable[InspectionRow],
    *,
    data_type: str,
) -> str:
    """DBやLLMを呼ばず、任意件数の履歴から集計文だけを作る。"""
    if not data_type:
        raise ValueError("data_type must not be empty.")

    records = []
    by_date = defaultdict(list)
    occurrences = defaultdict(list)
    for inspection in inspections:
        captured_at = inspection["captured_at"]
        try:
            timestamp = datetime.fromisoformat(str(captured_at))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid captured_at: {captured_at!r}") from exc
        # 元の履歴は変更せず、不良種別未記録のNGも集計から落とさない。
        record = {
            "captured_at": timestamp.isoformat(),
            "inspection_result": inspection["inspection_result"],
            "defect_type": str(inspection["defect_type"] or "unknown"),
        }
        records.append(record)
        by_date[timestamp.date().isoformat()].append(record)
        if record["inspection_result"] == "NG":
            occurrences[record["defect_type"]].append(timestamp)

    if not records:
        return "分析対象の検査記録はありません。"

    overall = calculate_quality_metrics(records)
    # 主要3種の順序を固定し、他の種別は実データから追加する。
    defect_keys = [key for key in DEFECT_LABELS if key in occurrences]
    defect_keys += sorted(set(occurrences) - set(DEFECT_LABELS))
    defect_names = [DEFECT_LABELS.get(key, key) for key in defect_keys]
    observed_dates = sorted(by_date)
    bands = calculate_time_band_quality_metrics(records)
    metric_rows = [("全体", overall)]
    metric_rows += [
        (day, calculate_quality_metrics(by_date[day])) for day in observed_dates
    ]
    metric_rows += [(TIME_BAND_LABELS[row["time_band"]], row) for row in bands]

    header = ["区分", "検査件数", "OK件数", "NG件数", "不良率(NG/検査)"]
    for name in defect_names:
        header += [name + "件数", name + "/検査", name + "/NG"]
    table = [header]
    for label, metrics in metric_rows:
        total, ng = metrics["total_inspections"], metrics["ng_count"]
        row = [label, total, metrics["ok_count"], ng, _ratio(ng, total)]
        for key in defect_keys:
            count = metrics["defect_counts"].get(key, 0)
            row += [count, _ratio(count, total), _ratio(count, ng)]
        table.append(row)

    cross_table = [
        ["日付・時間帯", "検査件数", "NG件数", "不良率(NG/検査)"]
        + [name + "件数" for name in defect_names]
    ]
    for day in observed_dates:
        for metrics in calculate_time_band_quality_metrics(by_date[day]):
            total, ng = metrics["total_inspections"], metrics["ng_count"]
            cross_table.append(
                [day + " " + TIME_BAND_LABELS[metrics["time_band"]], total, ng,
                 _ratio(ng, total)]
                + [metrics["defect_counts"].get(key, 0) for key in defect_keys]
            )

    # 複数欠品・数量異常・未記録等を、単一部品の欠品と説明しない。
    single_missing = bool(defect_keys) and all(
        key in ("washer_missing", "washer_before_spacer_missing", "washer_after_spacer_missing",
                "spacer_missing", "cap_nut_missing", "bolt_missing")
        for key in defect_keys
    )
    category = "欠品" if single_missing else "不良"
    occurrence_table = [[
        category + "種別", "件数", "初回記録", "最終記録", "発生日",
        "最終発生日後の観測日(発生0件)", "後続観測日数",
    ]]
    for key, name in zip(defect_keys, defect_names):
        found = sorted(occurrences[key])
        last_date = found[-1].date().isoformat()
        later_dates = [day for day in observed_dates if day > last_date]
        occurrence_table.append([
            name, len(found), found[0].isoformat(), found[-1].isoformat(),
            "・".join(sorted({value.date().isoformat() for value in found})),
            "・".join(later_dates) or "後続観測日なし", len(later_dates),
        ])

    share_table = [["時間帯", category + "種別", "当該時間帯の件数/全期間の同種" + category + "件数"]]
    for band in bands:
        for key, name in zip(defect_keys, defect_names):
            share_table.append([
                TIME_BAND_LABELS[band["time_band"]], name,
                _ratio(band["defect_counts"].get(key, 0), len(occurrences[key])),
            ])

    timestamps = [datetime.fromisoformat(row["captured_at"]) for row in records]
    data_description = (
        "分析機能検証用のダミー組付け検査記録。"
        if data_type == "test_data" else f"組付け検査記録（{data_type}）。"
    )
    # 同梱サンプルの入力では、データ種別の説明を省略する。
    data_prefix = "" if data_type == "portfolio_demo" else f"データ：{data_description}\n"
    classification = (
        "このデータではNG1件につき欠品種別1つ。"
        if single_missing else
        "不良種別はNGレコードの分類で、複数部品欠品も1件の分類として数える。"
    )
    return (
        data_prefix + f"観測範囲：{min(timestamps).isoformat()}〜{max(timestamps).isoformat()}、"
        f"記録のある日数：{len(observed_dates)}日、検査{len(records)}件。\n"
        f"OK=正常、NG=不良。{classification}作業者・設備・部品ロット・原因の記録は含まない。\n"
        "割合はPython計算済みの『分子件数/分母件数=割合』。"
        f"検査に対する{category}率とNG内の構成比は別の列。"
        "各行の検査・NGが分母。0件の分母は算出不可であり0%ではない。日付は記録の暦日。\n\n"
        "全体・日別・時間帯別集計(CSV)\n" + _csv_text(table)
        + "\n日付×時間帯集計(CSV)\n" + _csv_text(cross_table)
        + f"\n{category}の観測記録(CSV)\n" + _csv_text(occurrence_table)
        + f"\n{category}種別ごとの時間帯構成(CSV)\n" + _csv_text(share_table)
        + _dashboard_summary(overall, bands)
    )
