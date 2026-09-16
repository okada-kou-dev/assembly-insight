from collections.abc import Callable
from pathlib import Path

from src.inspection.batch import (
    find_image_paths,
    inspect_images,
)
from src.inspection.metadata import parse_captured_at
from src.inspection.persistence import save_image_inspection
from src.inspection.service import ImageInspection
from src.vision.detector import PartDetector


def inspect_directory_and_save(
    detector: PartDetector,
    image_dir: str | Path,
    db_path: str | Path,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    *,
    image_started_callback: Callable[[int, int, Path], None] | None = None,
    image_result_callback: Callable[[int, int, ImageInspection], None] | None = None,
    save_progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[ImageInspection]:
    """
    フォルダ内の画像を一括検査し、
    検査結果をSQLiteへ保存する。

    画像検査を全件終えてから保存する既存順序を維持する。
    保存通知は各保存の直前と成功後に、確認済み件数と対象画像を渡す。
    """
    image_paths = find_image_paths(image_dir)

    if not image_paths:
        raise ValueError(f"No images found in: {image_dir}")

    # YOLO推論を始める前に、
    # 全ファイル名の日時形式を検証する。
    for image_path in image_paths:
        parse_captured_at(image_path)

    inspections = inspect_images(
        detector=detector,
        image_paths=image_paths,
        progress_callback=progress_callback,
        image_started_callback=image_started_callback,
        image_result_callback=image_result_callback,
    )

    for saved, inspection in enumerate(inspections, start=1):
        if save_progress_callback is not None:
            save_progress_callback(saved - 1, len(inspections), inspection.image_path)

        save_image_inspection(
            db_path=db_path,
            inspection=inspection,
        )

        if save_progress_callback is not None:
            save_progress_callback(saved, len(inspections), inspection.image_path)

    return inspections
