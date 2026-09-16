import re
from datetime import datetime
from pathlib import Path

FILENAME_PATTERN = re.compile(r"^(?P<timestamp>\d{8}_\d{6})_.+$")


def parse_captured_at(image_path: str | Path) -> datetime:
    """
    画像ファイル名から撮影日時を取得する。

    Expected format:
        YYYYMMDD_HHMMSS_*.jpg

    Example:
        20260901_080000_001.jpg
        -> datetime(2026, 9, 1, 8, 0, 0)
    """
    image_path = Path(image_path)

    match = FILENAME_PATTERN.match(image_path.stem)

    if match is None:
        raise ValueError(
            "Invalid image filename format: "
            f"{image_path.name}. "
            "Expected YYYYMMDD_HHMMSS_*.jpg"
        )

    timestamp_text = match.group("timestamp")

    try:
        return datetime.strptime(
            timestamp_text,
            "%Y%m%d_%H%M%S",
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid captured datetime in filename: {image_path.name}"
        ) from exc
