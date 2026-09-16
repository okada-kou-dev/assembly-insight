"""配布用資産から起動するローカル版。生成条件は共通LLMクライアントを使う。"""

from pathlib import Path
import sys
from uuid import uuid4

# 配布先ルートと、このプロジェクト内の原稿置場の両方で同じ入口を使う。
ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.asset_catalog import load_catalog
from src.analysis.service import analyze_records
from src.llm.client import DEFAULT_MODEL, generate_free_quality_comment
from src.storage.database import list_inspections
from src.ui.display_text import display_text
from src.ui.inspection_panel import INSPECTION_REQUIRED_MESSAGE, has_completed_inspection, render_batch_inspection
from src.ui.quality_dashboard import render_ai_comment, render_quality_dashboard
from src.ui.theme import apply_theme, render_brand, section_label


def main():
    st.set_page_config(page_title="ASSEMBLY INSIGHT", layout="wide")
    apply_theme()
    render_brand()
    try:
        catalog = load_catalog(ROOT / "assets")
    except (ValueError, OSError, KeyError, TypeError):
        st.error("同梱の画像と画像AIモデルを確認してください")
        return
    st.session_state.setdefault("public_batch_db", str(ROOT / "outputs" / f"inspection_{uuid4().hex}.db"))
    st.session_state.setdefault("quality_db_path", st.session_state["public_batch_db"])
    st.session_state.setdefault("analysis_result", None)
    inspection_tab, quality_tab = st.tabs(["検査ワークスペース", "品質ダッシュボード"])
    with inspection_tab:
        section_label("INSPECTION WORKSPACE")
        render_batch_inspection(model_path=catalog.model_path,
                                default_input_dir=catalog.samples[0].path.parent,
                                default_db_path=Path(st.session_state["public_batch_db"]),
                                detection_output_dir=ROOT / "outputs/detections",
                                detector_settings=catalog.settings)
    with quality_tab:
        section_label("QUALITY ANALYTICS")
        st.header("品質ダッシュボード")
        db_text = st.text_input("品質分析対象のSQLite DB", key="quality_db_path")
        run = st.button("品質分析を実行", type="primary", key="public_quality_run")
        if not has_completed_inspection(db_text):
            st.session_state["analysis_result"] = None
            st.session_state.pop("public_analysis_error", None)
            if run:
                st.info(INSPECTION_REQUIRED_MESSAGE)
        elif run:
            st.session_state.pop("public_analysis_error", None)
            st.session_state["analysis_result"] = None
            try:
                if not Path(db_text).is_file():
                    raise FileNotFoundError("検査を実行し、分析対象のDBを確認してください")
                records = list_inspections(db_text)
                bundled_paths = {sample.path for sample in catalog.samples}
                data_type = ("portfolio_demo" if records and all(Path(row["image_path"]).resolve() in bundled_paths
                                                               for row in records) else "test_data")
                st.session_state["analysis_result"] = analyze_records(
                    records, source_db_path=db_text, data_type=data_type, generate_comment=False,
                )
            except Exception as exc:
                st.error(display_text(f"品質分析に失敗しました: {exc}"))
        result = st.session_state["analysis_result"]
        if result is not None:
            render_quality_dashboard(result, show_comment=False)
            with st.container(border=True):
                section_label("LOCAL AI / QUALITY COMMENT")
                st.subheader("ローカルAI分析")
                st.caption("AIが検査結果を読み解き、品質傾向の評価・考察と確認すべき点をコメントします")
                if run and result["overall"]["total_inspections"]:
                    try:
                        with st.spinner("AIが品質傾向を分析し、コメントを作成しています", show_time=True):
                            result["llm_comment"] = generate_free_quality_comment(result["llm_context"], model=DEFAULT_MODEL)
                    except Exception as exc:
                        st.session_state["public_analysis_error"] = display_text(f"AIコメントの生成に失敗しました: {exc}")
                if st.session_state.get("public_analysis_error"):
                    st.error(st.session_state["public_analysis_error"])
                render_ai_comment(result)


if __name__ == "__main__":
    main()
