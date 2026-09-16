"""保存された分析結果のダッシュボード。表示操作で分析を再実行しない。"""

from typing import Any

import streamlit as st
from src.ui.display_text import comment_markdown, display_lines, display_text

from src.ui.quality_charts import (
    confidence_chart, daily_chart, defect_rows, donut_chart, heatmap_chart,
    pareto_chart, residual_chart, volume_chart,
)
from src.ui.theme import section_label


def _chart(chart: Any) -> None:
    st.altair_chart(chart, width="stretch", theme=None)


def render_quality_dashboard(result: dict[str, Any], *, show_comment: bool = True) -> None:
    """表示用の構成比以外の品質数値は既存分析結果からそのまま参照する。"""
    overall = result["overall"]
    total = overall["total_inspections"]
    daily = result.get("daily", [])
    bands = result.get("time_bands")
    defects = defect_rows(overall)

    if result.get("source_db_path"):
        st.caption(display_text(f"表示中の分析対象DB：{result['source_db_path']}"))
    if daily:
        st.caption(display_text(f"観測範囲 {daily[0]['date']} — {daily[-1]['date']}  ·  記録のある日数 {len(daily)}日"))

    section_label("QUALITY OVERVIEW")
    columns = st.columns(4)
    for column, label, value in zip(columns, ["総検査数", "OK", "NG", "不良率"], [
        f"{total:,}", f"{overall['ok_count']:,}", f"{overall['ng_count']:,}",
        f"{overall['defect_rate']:.1%}" if total else "—",
    ]):
        column.metric(label, value)
    st.caption(display_text(f"不良率：NG {overall['ng_count']}件 / 検査 {total}件"
               if total else "検査記録0件。不良率は算出できません。"))

    left, right = st.columns([1.35, 1], gap="medium")
    with left, st.container(border=True):
        section_label("01 / QUALITY TREND")
        st.subheader("検査量と不良率の推移")
        if daily:
            _chart(daily_chart(daily))
            _chart(volume_chart(daily))
            st.caption(display_lines("上段：不良率。下段：OK・NG件数。"))
        else:
            st.info(display_text("表示できる品質履歴がありません。"))
    with right, st.container(border=True):
        section_label("02 / DEFECT COMPOSITION")
        st.subheader("不良構成比")
        if defects:
            _chart(donut_chart(defects))
            st.caption(display_lines("分母は全NG件数。"))
        else:
            st.info(display_text("不良データはありません。"))

    left, right = st.columns([1, 1.35], gap="medium")
    with left, st.container(border=True):
        section_label("03 / DEFECT RANKING")
        st.subheader("不良の内訳・優先確認")
        if defects:
            _chart(pareto_chart(defects))
            first = defects[0]
            st.caption(display_lines(f"最多分類：{first['defect']}  ·  {first['count']} / NG {overall['ng_count']}件"
                       f"（{first['share']:.1%}）。"))
        else:
            st.info(display_text("不良データはありません。"))
    with right, st.container(border=True):
        section_label("04 / DEFECT MATRIX")
        st.subheader("時間帯 × 不良種別")
        if bands and defects:
            _chart(heatmap_chart(bands, defects))
            st.caption(display_lines("セルの数字：該当NG件数。色：該当NG / その時間帯の検査件数。"))
        else:
            st.info(display_text("比較できる時間帯別の不良データはありません。"))

    left, right = st.columns([1, 1.35], gap="medium")
    with left, st.container(border=True):
        section_label("05 / INTERVAL ESTIMATION")
        st.subheader("時間帯別不良率と推定幅")
        intervals = result.get("confidence_intervals")
        if intervals:
            _chart(confidence_chart(intervals))
            st.caption(display_lines("点：実測不良率。線：95% Wilson信頼区間。少数の記録ほど推定幅が広がります。"))
            st.caption(display_lines("表示値はNG / 検査件数。各区間は開始時刻を含み、終了時刻は次の区間に入ります。"))
        else:
            st.info(display_text("時間帯別集計はありません。"))
    with right, st.container(border=True):
        section_label("06 / RESIDUAL DIAGNOSTICS")
        st.subheader("不良構成の残差分析")
        residuals = result.get("residual_analysis", {})
        if residuals.get("available"):
            st.caption(display_lines(f"Pearson χ² {residuals['chi_square']:.2f}  ·  自由度 {residuals['degrees_of_freedom']}"
                       f"  ·  対象NG {residuals['ng_total']}件"))
            _chart(residual_chart(residuals))
            st.caption(display_lines("1点＝時間帯×不良種別。上側・金色ほど期待より多く、下側・紫色ほど少ない組合せ。点の大きさはχ²への寄与。"))
            st.caption(display_lines("下の時間帯をクリックして選択・解除（複数選択可）。統計値は全時間帯の集計です。"))
            st.caption(display_lines("NG内の構成を比較しています。時間帯ごとの検査不良率とは分母が異なります。"))
            if residuals["small_expected_cells"]:
                st.caption(display_lines(f"期待件数5未満：{residuals['small_expected_cells']} / {len(residuals['cells'])}組。少数標本のため偏りの可視化として参照してください。"))
        else:
            st.info(display_text("残差分析には、NGがある時間帯と不良種別がそれぞれ2つ以上必要です。"))

    if show_comment:
        with st.container(border=True):
            section_label("LOCAL AI / QUALITY COMMENT")
            st.subheader("ローカルAI分析")
            st.caption(display_text("AIが検査結果を読み解き、品質傾向の評価・考察と確認すべき点をコメントします"))
            render_ai_comment(result)

    with st.expander("統計図の読み方と計算方法"):
        st.markdown(display_lines("**推定幅**：各時間帯のNGを二項比率として、両側95% Wilson区間を計算しています。"
                    "検査記録0件では推定しません。検査が独立で同じ条件から得られるという前提での参考値です。"))
        st.markdown(display_lines("**残差分析**：NGのみを対象とし、時間帯と不良種別が独立だと仮定した件数との差を表示します。"
                    "期待件数 E＝時間帯のNG数×種別のNG数÷全NG数、Pearson残差＝(実件数−E)÷√E、"
                    "χ²＝残差の二乗の合計です。NGが0件の行・列は除外します。p値や有意差判定は表示しません。"))


def render_ai_comment(result: dict[str, Any]) -> None:
    comment = result["llm_comment"]
    if comment is None:
        if not result["overall"]["total_inspections"]:
            st.info(display_text("分析対象データがありません"))
    elif isinstance(comment, str):
        st.markdown(comment_markdown(comment))
