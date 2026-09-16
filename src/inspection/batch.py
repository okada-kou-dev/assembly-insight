from collections.abc import Callable
from pathlib import Path

from src.inspection.service import ImageInspection, inspect_image
from src.vision.detector import PartDetector

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def find_image_paths(directory: str | Path) -> list[Path]:
    """
    指定フォルダ直下の画像ファイルを名前順で取得する。
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    image_paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]

    return sorted(
        image_paths,
        key=lambda path: path.name.lower(),
    )


def inspect_images(
    detector: PartDetector,
    image_paths: list[Path],
    progress_callback: Callable[[int, int, Path], None] | None = None,
    *,
    image_started_callback: Callable[[int, int, Path], None] | None = None,
    image_result_callback: Callable[[int, int, ImageInspection], None] | None = None,
) -> list[ImageInspection]:
    """画像を各1回検査する。任意通知は表示側へ渡し、既存進捗は検査完了を表す。"""
    inspections = []

    total = len(image_paths)

    for completed, image_path in enumerate(
        image_paths,
        start=1,
    ):
        if image_started_callback is not None:
            image_started_callback(completed, total, image_path)

        inspection = inspect_image(
            detector=detector,
            image_path=image_path,
        )

        inspections.append(inspection)

        if image_result_callback is not None:
            image_result_callback(completed, total, inspection)

        if progress_callback is not None:
            progress_callback(
                completed,
                total,
                image_path,
            )

    return inspections
