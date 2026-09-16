"""同梱100枚の数量・判定・欠品側を照合する。学習やLLM生成は行わない。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
from pathlib import Path
from time import perf_counter

from training.data import (CLASSES, STATE_COUNTS, STATE_DEFECTS, file_hash,
                           fresh_output, relative_file, write_new_json)
from training.metrics import check_acceptance, serialize_inspection, summarize

ROOT = Path(__file__).resolve().parents[1]


def load_samples(root):
    root = Path(root).resolve()
    config = json.loads((root / "assets/catalog.json").read_text(encoding="utf-8"))
    expected = json.loads((root / "training/demo_expected.json").read_text(encoding="utf-8"))["records"]
    by_id = {r["id"]: r for r in expected}
    if len(expected) != 100 or len(by_id) != 100 or len(config["samples"]) != 100:
        raise ValueError("100 distinct demo samples required")
    if {r["id"] for r in config["samples"]} != set(by_id):
        raise ValueError("demo identities differ")
    samples = []
    for asset in config["samples"]:
        row = by_id[asset["id"]]
        image = relative_file(root / "assets", asset["path"])
        if (image.name != row["filename"] or asset["view"] != row["direction"]
                or file_hash(image) != asset["sha256"]):
            raise ValueError("demo image identity differs")
        if (row["expected_result"] != ("OK" if row["state"] == "normal" else "NG")
                or row["expected_defect_type"] != STATE_DEFECTS[row["state"]]):
            raise ValueError("demo expectation differs from the state")
        samples.append({"id": row["id"], "direction": row["direction"], "state": row["state"],
                        "split": "demo_acceptance", "image": image.relative_to(root).as_posix(),
                        "sha256": asset["sha256"], "captured_at": row["scenario_timestamp"]})
    return config, samples


def evaluate(root, output, detector_factory=None):
    from src.inspection.service import inspect_image
    from src.vision.filtered_detector import DetectorSettings, FilteredPartDetector

    root = Path(root).resolve()
    output = fresh_output(root, output)
    config, samples = load_samples(root)
    weights = relative_file(root / "assets", config["model"]["path"])
    if file_hash(weights) != config["model"]["sha256"]:
        raise ValueError("bundled model changed")
    settings = DetectorSettings(**config["settings"])
    output.mkdir(parents=True, exist_ok=False)
    detector = (detector_factory or FilteredPartDetector)(weights, settings)
    if detector_factory is None:
        detector.model.to("cpu")
    records, raw_records = [], []
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for sample in samples:
            path = root / sample["image"]
            start = perf_counter()
            inspection = inspect_image(detector, path)
            record = serialize_inspection(sample, inspection, sample["sha256"], perf_counter() - start)
            record["captured_at"] = sample["captured_at"]
            check_acceptance(record)
            summarize([record])
            if file_hash(path) != sample["sha256"]:
                raise ValueError("demo image changed during inference")
            records.append(record)
            raw = detector.last_raw
            raw_records.append({"id": sample["id"], "image": sample["image"],
                                "image_sha256": sample["sha256"], "orig_shape": list(raw.orig_shape),
                                "boxes": raw.boxes.data.tolist()})
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    if file_hash(weights) != config["model"]["sha256"]:
        raise ValueError("model changed during inference")
    kind = "injected_test_detector" if detector_factory else "live_inference"
    result = {"kind": kind, "model_sha256": config["model"]["sha256"], "settings": settings.to_dict(),
              "matches": sum(r["all_matches"] for r in records), "total": len(records),
              "summary": summarize(records), "records": records,
              "notes": "Demo images were also used for tuning; this is not an independent accuracy estimate."}
    write_new_json(output / "raw_detections.json", {"model_sha256": result["model_sha256"],
                                                   "settings": settings.to_dict(), "records": raw_records})
    write_new_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    fresh_output(ROOT, args.output)
    _, samples = load_samples(ROOT)
    if args.run:
        result = evaluate(ROOT, args.output)
        print(f"Demo matches: {result['matches']}/{result['total']}; visual review is separate.")
    else:
        print(f"Checked {len(samples)} demo inputs. No inference or output creation.")


if __name__ == "__main__":
    main()
