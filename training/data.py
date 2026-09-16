"""公開データの照合と、学習用ディレクトリの新規作成。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PureWindowsPath
import shutil

CLASSES = ("bolt", "washer", "spacer", "cap_nut")
DIRECTIONS = ("front", "right", "left")
STATE_COUNTS = {
    "normal": (1, 2, 1, 1),
    "washer_front_missing": (1, 1, 1, 1),
    "washer_rear_missing": (1, 1, 1, 1),
    "spacer_missing": (1, 2, 0, 1),
    "cap_nut_missing": (1, 2, 1, 0),
}
STATE_DEFECTS = {
    "normal": None,
    "washer_front_missing": "washer_before_spacer_missing",
    "washer_rear_missing": "washer_after_spacer_missing",
    "spacer_missing": "spacer_missing",
    "cap_nut_missing": "cap_nut_missing",
}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def relative_file(root, value):
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise ValueError("relative path inside the supplied root required")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"missing or out-of-root input: {value}")
    return path


def fresh_output(root, value):
    path = (Path(root) / value).resolve()
    outputs = (Path(root) / "outputs").resolve()
    if not path.is_relative_to(outputs) or path == outputs:
        raise ValueError("output must be a new directory below outputs/")
    if path.exists():
        raise FileExistsError(path)
    return path


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def label_counts(path):
    counts = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"YOLO label must contain class x y width height: {path.name}")
        class_id = int(parts[0])
        values = tuple(float(v) for v in parts[1:])
        if class_id not in range(4) or any(not isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"invalid normalized label: {path.name}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"empty label box: {path.name}")
        counts[class_id] += 1
    return tuple(counts[i] for i in range(4))


def verify_dataset(root):
    """学習80枚を全件照合する。モデルを読み込まず、ファイルを書かない。"""
    root = Path(root).resolve()
    manifest = json.loads((root / "training/data/manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    expected = {}
    for direction in DIRECTIONS:
        for state in STATE_COUNTS:
            for n in range(1, 11 if direction == "front" else 4):
                expected[f"{direction}_{state}_{n:03d}"] = (
                    direction, state, "train" if n <= (8 if direction == "front" else 2) else "val")
    if len(records) != 80 or {r["id"] for r in records} != set(expected):
        raise ValueError("80 distinct training image identities required")
    seen = set()
    for row in records:
        if (row["direction"], row["state"], row["split"]) != expected[row["id"]]:
            raise ValueError("training split or category differs")
        for key in ("image", "label"):
            path = relative_file(root / "training", row[key])
            if not path.is_relative_to(root / "training/data" / ("images" if key == "image" else "labels")):
                raise ValueError("training inputs must be in their asset directory")
            if path in seen or file_hash(path) != row[key + "_sha256"]:
                raise ValueError(f"duplicate or changed training asset: {row[key]}")
            seen.add(path)
        if label_counts(root / "training" / row["label"]) != STATE_COUNTS[row["state"]]:
            raise ValueError("label class counts differ from the sample state")
    actual = {p.resolve() for folder in ("images", "labels")
              for p in (root / "training/data" / folder).rglob("*") if p.is_file()}
    if actual != seen:
        raise ValueError("extra or missing training asset")
    return records


def select_records(records, stage):
    if stage not in ("baseline", "finetune"):
        raise ValueError("unknown training stage")
    selected = [r for r in records if stage == "finetune" or r["direction"] == "front"]
    expected = {"train": 40, "val": 10} if stage == "baseline" else {"train": 60, "val": 20}
    if Counter(r["split"] for r in selected) != expected:
        raise ValueError("training/validation allocation differs")
    return selected


def prepare_dataset(root, records, output):
    """既存データを変更せず、検証済みYOLOラベルと画像を複製する。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root / "outputs"):
        raise ValueError("new dataset below outputs/ required")
    output.mkdir(parents=True, exist_ok=False)
    for row in records:
        for key, folder in (("image", "images"), ("label", "labels")):
            source = relative_file(root / "training", row[key])
            if file_hash(source) != row[key + "_sha256"]:
                raise ValueError("training input changed before copying")
            target = output / folder / row["split"] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if file_hash(target) != row[key + "_sha256"]:
                raise ValueError("training copy hash differs")
    config = output / "dataset.yaml"
    # JSON is a YAML subset and avoids adding a separate YAML-writing dependency.
    write_new_json(config, {"path": output.as_posix(), "train": "images/train",
                            "val": "images/val", "names": list(CLASSES)})
    return config
