"""同梱画像・モデル・推論設定の読込み。"""
from dataclasses import dataclass
import json
from pathlib import Path
from src.inspection.metadata import parse_captured_at
from src.vision.filtered_detector import DetectorSettings

ROOT = Path(__file__).resolve().parents[1] / "assets"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    view: str
    sha256: str


@dataclass(frozen=True)
class Catalog:
    model_path: Path
    model_sha256: str
    settings: DetectorSettings
    samples: tuple[Sample, ...]


def load_catalog(root: Path = ROOT) -> Catalog:
    root = root.resolve()
    config = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("synthetic_datetime_labels") is not True:
        raise ValueError("サンプル定義を確認してください")

    def asset(value: str) -> Path:
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("同梱資産を確認してください")
        return path

    samples = tuple(Sample(row["id"], asset(row["path"]), row["view"], row["sha256"])
                    for row in config["samples"])
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("サンプルIDが重複しているか、サンプルがありません")
    if len({sample.path for sample in samples}) != len(samples):
        raise ValueError("同じ画像が重複しています")
    for sample in samples:
        parse_captured_at(sample.path)
    return Catalog(asset(config["model"]["path"]), config["model"]["sha256"],
                   DetectorSettings(**config["settings"]), samples)
