"""Python集計済みの値を図へ写す。DB取得・判定・LLM生成は行わない。"""

from typing import Any

import altair as alt
import pandas as pd

from src.analysis.comment_summary import DEFECT_LABELS, TIME_BAND_LABELS
from src.analysis.statistics import ng_category_counts

CYAN = "#38BDF8"
ROSE = "#FB7185"
BLUE = "#4077B5"
MUTED = "#9BAEC7"
TEXT = "#E6EDF7"
BAND_LABELS = list(TIME_BAND_LABELS.values())
DEFECT_COLORS = {
    "washer_before_spacer_missing": "#38BDF8", "washer_after_spacer_missing": "#A78BFA",
    "cap_nut_missing": "#FB7185", "spacer_missing": "#FBBF24", "bolt_missing": "#2DD4BF",
    "washer_missing": "#8FA5BF", "multiple_missing": "#F97316",
    "quantity_mismatch": "#E879F9", "unknown": "#65758B",
}


def _defect_color(rows: list[dict], *, legend: Any = None) -> alt.Color:
    return alt.Color("defect:N", sort=None, legend=legend,
                     scale=alt.Scale(domain=[row["defect"] for row in rows],
                                     range=[DEFECT_COLORS.get(row["key"], MUTED) for row in rows]))


def _finish(chart: Any) -> Any:
    return (chart.configure(background="transparent", font="Segoe UI, Meiryo, sans-serif")
            .configure_view(strokeOpacity=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#233249",
                            domain=False, tickColor="#30415A", labelFontSize=11,
                            titleFontSize=11, titlePadding=12, labelPadding=8)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=11,
                              orient="top", title=None)
            .configure_title(color=TEXT, anchor="start", fontSize=12))


def defect_rows(metrics: dict) -> list[dict]:
    """種別未記録のNGも含める。構成比の分母は全NG件数。"""
    counts = ng_category_counts(metrics)
    rows, cumulative = [], 0
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if not count:
            continue
        cumulative += count
        rows.append({"key": key, "defect": DEFECT_LABELS.get(key, key), "count": count,
                     "share": count / metrics["ng_count"],
                     "cumulative": cumulative / metrics["ng_count"],
                     "ng_total": metrics["ng_count"]})
    return rows


def daily_chart(daily: list[dict]) -> Any:
    """観測日のみを描く。件数と率は上下で分け、率の軸は0〜100%。"""
    frame = pd.DataFrame(daily)
    base = alt.Chart(frame).encode(x=alt.X("date:O", title=None, sort=None,
                                        axis=alt.Axis(labelAngle=-25)))
    tips = [alt.Tooltip("date:O", title="撮影日"),
            alt.Tooltip("total_inspections:Q", title="検査件数", format=",d"),
            alt.Tooltip("ng_count:Q", title="NG件数", format=",d"),
            alt.Tooltip("defect_rate:Q", title="不良率 NG / 検査", format=".1%")]
    rate = base.mark_line(color=CYAN, strokeWidth=3, point=alt.OverlayMarkDef(size=65)).encode(
        y=alt.Y("defect_rate:Q", title="不良率", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", tickCount=5)), tooltip=tips)
    return _finish(rate.properties(height=175))


def volume_chart(daily: list[dict]) -> Any:
    base = alt.Chart(pd.DataFrame(daily)).encode(
        x=alt.X("date:O", title=None, sort=None, axis=alt.Axis(labelAngle=-25)))
    volume = base.transform_fold(["ok_count", "ng_count"], as_=["outcome", "count"]).mark_bar(
        cornerRadiusTopLeft=2, cornerRadiusTopRight=2, size=22).encode(
            y=alt.Y("count:Q", title="OK・NG件数", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("outcome:N", title=None,
                            scale=alt.Scale(domain=["ok_count", "ng_count"], range=[BLUE, ROSE]),
                            legend=alt.Legend(labelExpr="datum.label === 'ok_count' ? 'OK' : 'NG'")),
            tooltip=[alt.Tooltip("date:O", title="撮影日"),
                     alt.Tooltip("total_inspections:Q", title="検査件数"),
                     alt.Tooltip("ok_count:Q", title="OK件数"), alt.Tooltip("ng_count:Q", title="NG件数")])
    return _finish(volume.properties(height=160))


def pareto_chart(rows: list[dict]) -> Any:
    base = alt.Chart(pd.DataFrame(rows)).encode(
        y=alt.Y("defect:N", sort=None, title=None, axis=alt.Axis(labelLimit=230)),
        x=alt.X("count:Q", title="NG件数", scale=alt.Scale(zero=True),
                axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("defect:N", title="不良種別"),
                 alt.Tooltip("count:Q", title="件数"), alt.Tooltip("ng_total:Q", title="全NG件数"),
                 alt.Tooltip("share:Q", title="NG内構成比", format=".1%"),
                 alt.Tooltip("cumulative:Q", title="上位からの累積構成比", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, size=22).encode(color=_defect_color(rows))
    values = base.mark_text(align="left", dx=7, color=TEXT).encode(text="count:Q")
    return _finish((bars + values).properties(height=max(170, len(rows) * 40), padding={"right": 28}))


def time_band_rows(bands: list[dict]) -> list[dict]:
    return [{**row, "band": TIME_BAND_LABELS[row["time_band"]],
             "rate": row["defect_rate"] if row["total_inspections"] else None,
             "label": f"{row['defect_rate']:.1%}" if row["total_inspections"] else "記録なし"}
            for row in bands]


def donut_chart(rows: list[dict]) -> Any:
    base = alt.Chart(pd.DataFrame(rows))
    arcs = base.mark_arc(innerRadius=68, outerRadius=105, padAngle=0.025, cornerRadius=3).encode(
        theta=alt.Theta("count:Q", stack=True),
        order=alt.Order("count:Q", sort="descending"),
        color=_defect_color(rows, legend=alt.Legend(orient="bottom", columns=1, labelLimit=300,
                                                   rowPadding=6, symbolType="circle")),
        tooltip=[alt.Tooltip("defect:N", title="不良種別"), alt.Tooltip("count:Q", title="件数"),
                 alt.Tooltip("ng_total:Q", title="全NG件数"),
                 alt.Tooltip("share:Q", title="NG内構成比", format=".1%")])
    center = alt.Chart(pd.DataFrame([{"total": rows[0]["ng_total"]}]))
    number = center.mark_text(color=TEXT, fontSize=32, fontWeight=600, dy=-7).encode(text="total:Q")
    caption = center.mark_text(color=MUTED, fontSize=10, dy=22).encode(text=alt.value("TOTAL NG"))
    return _finish((arcs + number + caption).properties(height=370))


def confidence_chart(rows: list[dict]) -> Any:
    display_rows = [{**row, "label": f"{row['ng']} / {row['total']}" if row["total"] else "記録なし"}
                    for row in rows]
    base = alt.Chart(pd.DataFrame(display_rows)).encode(
        y=alt.Y("band:N", title=None, sort=BAND_LABELS),
        tooltip=[alt.Tooltip("band:N", title="時間帯"), alt.Tooltip("ng:Q", title="NG件数"),
                 alt.Tooltip("total:Q", title="検査件数"), alt.Tooltip("rate:Q", title="不良率", format=".1%"),
                 alt.Tooltip("lower:Q", title="95%区間 下限", format=".1%"),
                 alt.Tooltip("upper:Q", title="95%区間 上限", format=".1%")])
    axis = alt.X("lower:Q", title="不良率 / 95%信頼区間", scale=alt.Scale(domain=[0, 1]),
                 axis=alt.Axis(format=".0%", tickCount=5))
    intervals = base.mark_rule(color="#4077B5", strokeWidth=5).encode(x=axis, x2="upper:Q")
    points = base.mark_point(color=CYAN, filled=True, size=100).encode(x="rate:Q")
    labels = base.mark_text(color=TEXT, align="right", dx=-3, dy=-16, fontSize=11).encode(
        x=alt.datum(1), text="label:N")
    return _finish((intervals + points + labels).properties(height=300))


def heatmap_rows(bands: list[dict], defects: list[dict]) -> list[dict]:
    rows = []
    for band in bands:
        counts = {row["key"]: row["count"] for row in defect_rows(band)}
        total = band["total_inspections"]
        for defect in defects:
            count = counts.get(defect["key"], 0)
            rows.append({"band": TIME_BAND_LABELS[band["time_band"]], "defect": defect["defect"],
                         "count": count, "total": total, "rate": count / total if total else None,
                         "label": str(count) if total else "—"})
    return rows


def heatmap_chart(bands: list[dict], defects: list[dict]) -> Any:
    base = alt.Chart(pd.DataFrame(heatmap_rows(bands, defects))).encode(
        x=alt.X("band:N", title=None, sort=BAND_LABELS,
                axis=alt.Axis(labelAngle=-20, labelLimit=150)),
        y=alt.Y("defect:N", title=None, sort=[row["defect"] for row in defects],
                axis=alt.Axis(labelLimit=230)),
        tooltip=[alt.Tooltip("band:N", title="時間帯"), alt.Tooltip("defect:N", title="不良種別"),
                 alt.Tooltip("count:Q", title="該当NG件数"), alt.Tooltip("total:Q", title="時間帯の検査件数"),
                 alt.Tooltip("rate:Q", title="該当NG / 時間帯の検査", format=".1%")])
    cells = base.mark_rect(stroke="#111B2C", strokeWidth=3, cornerRadius=4, invalid=None).encode(
        color=alt.condition("datum.total > 0", alt.Color("rate:Q", title="該当NG / 検査",
                            scale=alt.Scale(domain=[0, 0.2, 0.5, 1],
                                            range=["#213653", "#287EAD", "#36C3B1", "#F5CB72"]),
                            legend=alt.Legend(format=".0%", gradientLength=110)), alt.value("#1A2434")))
    labels = base.mark_text(fontSize=13, fontWeight=600).encode(
        text="label:N", color=alt.condition("datum.rate > 0.45", alt.value("#0B1120"), alt.value(TEXT)))
    return _finish((cells + labels).properties(height=max(270, len(defects) * 46 + 100)))


def residual_chart(analysis: dict) -> Any:
    rows = [{**cell, "defect": DEFECT_LABELS.get(cell["key"], cell["key"])} for cell in analysis["cells"]]
    limit = max(2, max(abs(row["residual"]) for row in rows) * 1.15)
    selected_band = alt.selection_point(fields=["band"], bind="legend", toggle="true")
    base = alt.Chart(pd.DataFrame(rows)).encode(
        x=alt.X("expected:Q", title="独立を仮定した期待件数",
                scale=alt.Scale(domain=[0, max(row["expected"] for row in rows)]), axis=alt.Axis(tickCount=6)),
        y=alt.Y("residual:Q", title="Pearson残差", scale=alt.Scale(domain=[-limit, limit])),
        tooltip=[alt.Tooltip("band:N", title="時間帯"), alt.Tooltip("defect:N", title="不良種別"),
                 alt.Tooltip("observed:Q", title="実際の件数"), alt.Tooltip("expected:Q", title="期待件数", format=".2f"),
                 alt.Tooltip("residual:Q", title="Pearson残差", format="+.2f"),
                 alt.Tooltip("contribution:Q", title="χ²への寄与", format=".2f")])
    points = base.mark_point(filled=True, opacity=0.85, stroke="#111B2C", strokeWidth=1).encode(
        color=alt.Color("residual:Q", title="期待からの偏り",
                        scale=alt.Scale(domain=[-limit, 0, limit], range=["#A78BFA", "#4C7897", "#F5CB72"]),
                        legend=alt.Legend(gradientLength=120, format="+.1f")),
        size=alt.Size("contribution:Q", title="χ²への寄与",
                      scale=alt.Scale(domain=[0, max(row["contribution"] for row in rows) or 1], range=[45, 550]), legend=None),
        shape=alt.Shape("band:N", title="時間帯（クリックで絞込）",
                        scale=alt.Scale(domain=BAND_LABELS, range=["circle", "square", "diamond", "triangle-up"]),
                        legend=alt.Legend(orient="bottom", columns=2, labelLimit=130,
                                          symbolStrokeColor=MUTED, symbolFillColor=MUTED))
    ).add_params(selected_band).transform_filter(selected_band)
    zero = alt.Chart(pd.DataFrame([{"zero": 0}])).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="zero:Q")
    return _finish((zero + points).properties(height=345))
