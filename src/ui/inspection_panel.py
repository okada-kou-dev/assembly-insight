"""一括検査の操作・段階表示・セッション内の結果表示。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
from src.ui.display_text import display_text

from src.inspection.metadata import parse_captured_at
from src.inspection.persistence import get_representative_confidence
from src.inspection.service import ImageInspection
from src.inspection.workflow import inspect_directory_and_save
from src.storage.database import list_inspections
from src.storage.replacement import database_snapshot, publish_replacement
from src.ui.inspection_animation import (
    SPEED_PRESETS,
    AnimationSpeed,
    DisplayDetection,
    hold_decision,
    load_original_image,
    ordered_detections,
    play_detection_steps,
    quantity_comparison,
    render_detection_frame,
)
from src.vision.detector import PartDetector
from src.vision.filtered_detector import DetectorSettings, FilteredPartDetector
from src.ui.theme import empty_workspace

DEFECT_LABELS = {
    "washer_missing": "ワッシャ欠品（位置未確定）",
    "washer_before_spacer_missing": "ワッシャ欠品（ボルト側）",
    "washer_after_spacer_missing": "ワッシャ欠品（化粧ナット側）",
    "spacer_missing": "スペーサー欠品",
    "cap_nut_missing": "化粧ナット欠品",
    "bolt_missing": "ボルト欠品",
    "multiple_missing": "複数部品欠品",
    "quantity_mismatch": "数量異常",
}


INSPECTION_REQUIRED_MESSAGE = "先に「検査ワークスペース」で「一括検査を実行」を押し、検査を完了してください"


def has_completed_inspection(db_path: str) -> bool:
    """このセッションで全件の検査・保存を完了したDBだけを分析対象にする。"""
    completed = st.session_state.get("completed_inspection_db")
    if completed is None:
        return False
    try:
        return Path(db_path).resolve() == Path(completed)
    except (OSError, ValueError):
        return False


def defect_label(defect_type: str | None) -> str:
    return "-" if defect_type is None else DEFECT_LABELS.get(defect_type, defect_type)


@dataclass
class BatchDisplayState:
    db_path: str
    speed_name: str
    stage: str = "入力確認"
    active_path: str = ""
    captured_at: str = ""
    image_number: int = 0
    total: int = 0
    inspected: int = 0
    saved: int = 0
    saving_path: str = ""
    frame: np.ndarray | None = None
    detection_rows: list[dict] = field(default_factory=list)
    quantity_rows: list[dict] = field(default_factory=list)
    decision: str | None = None
    defect: str | None = None
    representative_confidence: float | None = None
    inspection_rows: list[dict] = field(default_factory=list)
    saved_rows: list[dict] = field(default_factory=list)
    previews: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    backup_path: str | None = None


class InspectionPanel:
    """同じ表示領域だけを更新し、描画中の仮の状態をDBへ渡さない。"""

    def __init__(self) -> None:
        self.heading = st.empty()
        self.status = st.empty()
        image_column, detail_column = st.columns([3, 2])
        with image_column:
            self.image = st.empty()
        with detail_column:
            self.detections = st.empty()
            self.quantities = st.empty()
            self.decision = st.empty()
            self.confidence = st.empty()
            self.position = st.empty()
        self.progress = st.empty()
        self.saving = st.empty()
        self.failure = st.empty()
        self.results = st.empty()

    def show_detection_frame(self, state: BatchDisplayState) -> None:
        """枠追加中は画像・部品一覧・ステータスだけを差し替える。"""
        self.status.info(display_text(state.stage))
        if state.frame is None:
            self.image.empty()
        else:
            self.image.image(state.frame, channels="BGR", width="stretch")
        if state.detection_rows:
            self.detections.dataframe(state.detection_rows, hide_index=True, width="stretch")
        else:
            self.detections.info(display_text("検出情報は結果取得後に表示します。"))

    def _show_live_image(self, state: BatchDisplayState) -> None:
        if state.active_path:
            self.heading.markdown(
                display_text(f"**画像 {state.image_number} / {state.total}：{Path(state.active_path).name}**  \n"
                f"撮影日時：{state.captured_at}")
            )
        else:
            self.heading.empty()
        self.show_detection_frame(state)
        if state.quantity_rows:
            self.quantities.dataframe(state.quantity_rows, hide_index=True, width="stretch")
        else:
            self.quantities.empty()
        if state.decision is None:
            self.decision.empty()
            self.confidence.empty()
        else:
            message = f"判定：{state.decision} / 不良内容：{defect_label(state.defect)}"
            if state.decision == "OK":
                self.decision.success(display_text(message))
            else:
                self.decision.error(display_text(message))
            value = state.representative_confidence
            self.confidence.caption(
                display_text("保存する代表confidence（全検出の最小値）："
                + ("算出なし（検出0件）" if value is None else f"{value:.3f}"))
            )

    def show_progress(self, state: BatchDisplayState) -> None:
        """画像や結果表を描画せず、検査済み件数とバーだけを更新する。"""
        percent = state.inspected / state.total if state.total else 0.0
        self.progress.progress(percent, text=f"画像検査済み {state.inspected} / {state.total} 件")

    def show(self, state: BatchDisplayState) -> None:
        if state.speed_name == "結果のみ":
            for area in (self.heading, self.image, self.detections, self.quantities,
                         self.decision, self.confidence):
                area.empty()
            self.status.info(display_text(state.stage if state.error or state.stage == "完了"
                                          else "一括検査中"))
        else:
            self._show_live_image(state)
        self.position.empty()
        self.show_progress(state)
        if state.stage == "完了":
            self.saving.success(display_text(f"DB保存完了：{state.saved} / {state.total} 件 / {state.db_path}"))
        elif state.saved or state.stage == "DB保存中":
            self.saving.info(
                display_text(f"DB保存確認済み {state.saved} / {state.total} 件"
                + (f" / 対象：{Path(state.saving_path).name}" if state.saving_path else ""))
            )
        else:
            self.saving.empty()
        if state.error:
            self.failure.error(
                display_text(f"一括検査の処理エラー（製品の検査NGとは別）：{state.stage} / "
                f"{state.saving_path or state.active_path or '入力確認'}\n\n{state.error}\n\n"
                f"画像検査済み {state.inspected} 件、DB保存確認済み {state.saved} 件。")
            )
        else:
            self.failure.empty()
        if state.inspection_rows:
            self.results.dataframe(state.inspection_rows, hide_index=True, width="stretch")
        else:
            self.results.empty()


class BatchPresenter:
    def __init__(
        self, state: BatchDisplayState, panel: InspectionPanel, preview_dir: Path,
        *, wait: Callable[[float], None] = sleep,
    ) -> None:
        self.state = state
        self.panel = panel
        self.preview_dir = preview_dir
        self.speed: AnimationSpeed = SPEED_PRESETS[state.speed_name]
        self.live_display = state.speed_name != "結果のみ"
        self.wait = wait

    def image_started(self, number: int, total: int, image_path: Path) -> None:
        state = self.state
        state.image_number, state.total = number, total
        state.active_path = str(image_path)
        state.captured_at = parse_captured_at(image_path).isoformat(sep=" ", timespec="seconds")
        state.stage = "画像読込中"
        state.frame = None
        state.detection_rows = []
        state.quantity_rows = []
        state.decision = None
        state.defect = None
        state.representative_confidence = None
        if self.live_display:
            self.panel.show(state)
            state.frame = render_detection_frame(load_original_image(image_path))
        elif number == 1:
            self.panel.show_progress(state)
        state.stage = "画像AI推論中"
        if self.live_display:
            self.panel.show(state)

    def image_result(self, number: int, total: int, inspection: ImageInspection) -> None:
        state = self.state
        # この時点では検査処理は完了している。判定の画面表示は枠表示の後に行う。
        state.inspected, state.total = number, total
        state.stage = "検出結果取得（数量照合の表示は全枠表示後）"
        if self.live_display:
            self.panel.show(state)

        def show_frame(frame: np.ndarray, visible: tuple[DisplayDetection, ...], count: int) -> None:
            state.frame = frame
            state.stage = (
                f"検出結果を順次表示中：{len(visible)} / {count} 枠"
                if self.speed.step_seconds > 0 else f"検出結果を表示：{count} 枠（演出なし）"
            )
            state.detection_rows = [
                {"表示済みの部品": item.label, "検出confidence": f"{item.confidence:.3f}"}
                for item in visible
            ]
            self.panel.show_detection_frame(state)

        if self.live_display:
            detections = play_detection_steps(inspection.model_result, self.speed, show_frame, wait=self.wait)
        else:
            # 表示用の途中フレームを作らず、保存画像用の検出情報だけを取得する。
            detections = ordered_detections(inspection.model_result)
        state.stage = "検査結果確定（DB未保存）"
        state.decision = inspection.decision.inspection_result
        state.defect = inspection.decision.defect_type
        state.representative_confidence = get_representative_confidence(inspection)
        if self.live_display:
            state.quantity_rows = quantity_comparison(inspection.counts)
            if not detections:
                state.detection_rows = [{"表示済みの部品": "検出0件", "検出confidence": "-"}]
            self.panel.show(state)
            hold_decision(self.speed, wait=self.wait)

        # 最終画像のみ保存する。全段階のフレームは保持しない。
        state.stage = "検出画像の保存中"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = self.preview_dir / f"{number:04d}_{inspection.image_path.stem}.jpg"
        final_frame = render_detection_frame(inspection.model_result.orig_img, detections)
        encoded, buffer = cv2.imencode(".jpg", final_frame)
        if not encoded:
            raise OSError(f"検出画像を作成できません: {preview_path}")
        with preview_path.open("xb") as preview_file:
            preview_file.write(buffer.tobytes())
        state.previews[str(inspection.image_path)] = str(preview_path)
        confidence = state.representative_confidence
        state.inspection_rows.append({
            "撮影日時": state.captured_at,
            "画像": inspection.image_path.name,
            "判定": state.decision,
            "不良内容": defect_label(state.defect),
            "代表confidence": "-" if confidence is None else f"{confidence:.3f}",
        })
        state.frame = final_frame if self.live_display else None
        state.stage = "検査結果確定（DB未保存）"
        if self.live_display:
            self.panel.show(state)

    def progress(self, completed: int, total: int, image_path: Path) -> None:
        self.state.inspected, self.state.total = completed, total
        if not self.live_display:
            self.panel.show_progress(self.state)

    def save_progress(self, saved: int, total: int, image_path: Path) -> None:
        self.state.stage = "DB保存中"
        self.state.saving_path = str(image_path)
        self.state.saved, self.state.total = saved, total
        if self.live_display:
            self.panel.show(self.state)


def _show_saved_images(state: BatchDisplayState) -> None:
    if not state.saved_rows:
        return
    with st.expander("画像別判定"):
        paths = [row["image_path"] for row in state.saved_rows]
        selected = st.selectbox("確認する画像", paths, format_func=lambda value: Path(value).name,
                                key=f"batch_gallery_{state.db_path}")
        row = next(row for row in state.saved_rows if row["image_path"] == selected)
        preview = state.previews.get(selected)
        if preview and Path(preview).is_file():
            st.image(preview, caption="保存済み検査の検出結果", width="stretch")
        elif Path(selected).is_file():
            st.image(selected, caption="元画像", width="stretch")
        else:
            st.warning(display_text("画像ファイルが見つかりません。"))
        st.write(display_text(f"判定：{row['inspection_result']} / 不良内容：{defect_label(row['defect_type'])}"))
        confidence = row["confidence"]
        st.caption(display_text("DBの代表confidence：" + ("-" if confidence is None else f"{confidence:.3f}")))


def render_batch_inspection(
    *, model_path: Path, default_input_dir: Path, default_db_path: Path,
    detection_output_dir: Path, detector_settings: DetectorSettings | None = None,
    protected_db_path: Path | None = None,
) -> None:
    st.header("一括検査")
    with st.container(border=True):
        input_column, db_column = st.columns(2)
        with input_column:
            input_text = st.text_input("検査画像フォルダ", value=str(default_input_dir))
        with db_column:
            db_text = st.text_input("検査結果の保存先SQLite DB", value=str(default_db_path))
        speed_name = st.radio("表示モード", list(SPEED_PRESETS), horizontal=True)
    run = st.button("一括検査を実行", type="primary", key="batch_run")
    request = (input_text, str(Path(db_text).resolve()), speed_name)
    pending = st.session_state.get('batch_overwrite')
    if pending and pending['request'] != request:
        st.session_state.pop('batch_overwrite', None)
        pending = None
    snapshot = None
    confirmation_area = st.empty()
    if run:
        st.session_state["completed_inspection_db"] = None
        st.session_state["analysis_result"] = None
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("public_analysis_error", None)
        st.session_state.pop('batch_overwrite', None)
        pending = None
        if Path(db_text).exists():
            try:
                if protected_db_path and Path(db_text).resolve() == protected_db_path.resolve():
                    raise ValueError("受入確認済みの原本DBです / 検査結果用の保存先を指定してください")
                pending = {'request': request, 'snapshot': database_snapshot(Path(db_text))}
                st.session_state['batch_overwrite'] = pending
            except (ValueError, OSError) as exc:
                st.error(display_text(str(exc)))
            run = False
    if pending:
        with confirmation_area.container():
            st.warning(display_text(f"保存先DBは既に存在します / 今回の検査結果で上書きしますか？\n\n{request[1]}\n\n元のDBは同じフォルダへバックアップします"))
            confirm = st.button("上書きして実行", key="batch_overwrite_confirm", type="primary")
            cancel = st.button("キャンセル", key="batch_overwrite_cancel")
        if cancel:
            st.session_state.pop('batch_overwrite', None)
            st.rerun()
        if confirm:
            snapshot = pending['snapshot']
            st.session_state.pop('batch_overwrite', None)
            run = True
            confirmation_area.empty()
    if not run and "batch_display_state" not in st.session_state:
        empty_workspace("検査の準備ができました。",
                        "画像フォルダと保存先DBを指定し、一括検査を開始してください。")
        return

    panel = InspectionPanel()
    if run:
        st.session_state["completed_inspection_db"] = None
        state = BatchDisplayState(db_path=db_text, speed_name=speed_name)
        st.session_state["batch_display_state"] = state
        try:
            image_dir, db_path = Path(input_text), Path(db_text)
            if not image_dir.is_dir():
                raise ValueError(f"画像フォルダが見つかりません: {image_dir}")
            if not model_path.is_file():
                raise ValueError(f"YOLOモデルが見つかりません: {model_path}")
            if snapshot:
                if database_snapshot(db_path) != snapshot:
                    raise ValueError("確認後に保存先DBが変更されました / 再度確認してください")
            elif db_path.exists():
                raise ValueError("保存先DBが作成されました / 再度実行して上書きを確認してください")
            working_db = (db_path.with_name(f'.{db_path.stem}.pending-{uuid4().hex}.db')
                          if snapshot else db_path)
            presenter = BatchPresenter(state, panel, detection_output_dir / uuid4().hex)
            state.stage = "画像AIモデル読み込み中"
            panel.show(state)
            detector = (FilteredPartDetector(model_path, detector_settings) if detector_settings is not None
                        else PartDetector(model_path=model_path, conf=0.25))
            state.stage = "入力画像・撮影日時の確認"
            inspect_directory_and_save(
                detector=detector, image_dir=image_dir, db_path=working_db,
                progress_callback=presenter.progress,
                image_started_callback=presenter.image_started,
                image_result_callback=presenter.image_result,
                save_progress_callback=presenter.save_progress,
            )
            state.stage = "保存結果の読込中"
            state.saved_rows = list_inspections(working_db)
            if snapshot:
                backup = publish_replacement(working_db, db_path, snapshot)
                state.backup_path = str(backup)
                st.session_state['batch_backup_path'] = str(backup)
            state.stage = "完了"
            if state.saved_rows:
                st.session_state["completed_inspection_db"] = str(db_path.resolve())
            st.session_state["quality_db_path"] = str(db_path)
            st.session_state["analysis_result"] = None
            st.session_state.pop('analysis_error', None)
        except Exception as exc:
            state.error = str(exc)
            if snapshot:
                state.error += " / 元の保存先DBは置き換えていません"
    else:
        state = st.session_state["batch_display_state"]
    panel.show(state)
    if state.backup_path:
        st.caption(display_text(f"上書き前のDBバックアップ：{state.backup_path}"))
    _show_saved_images(state)
