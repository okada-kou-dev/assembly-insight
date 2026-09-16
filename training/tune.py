"""保存した検出候補から全画像共通の設定を比較する。画像AIの再推論はしない。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.data import CLASSES, fresh_output, relative_file, write_new_json
from training.evaluate import load_samples
from training.metrics import check_acceptance, serialize_inspection, summarize

ROOT = Path(__file__).resolve().parents[1]


class CachedDetector:
    def __init__(self, raw, settings):
        self.raw, self.settings = raw, settings

    def predict(self, image_path):
        from src.vision.filtered_detector import selected_indices
        boxes = self.raw["boxes"]
        selected = [boxes[i] for i in selected_indices(boxes, self.settings)]
        return SimpleNamespace(names=dict(enumerate(CLASSES)),
            boxes=[SimpleNamespace(cls=np.asarray([b[5]]), conf=np.asarray([b[4]]), xyxy=np.asarray([b[:4]]))
                   for b in selected], orig_shape=tuple(self.raw["orig_shape"]), orig_img=None)


def replay(root, samples, raw_by_id, settings):
    from src.inspection.service import inspect_image
    records = []
    for sample in samples:
        inspection = inspect_image(CachedDetector(raw_by_id[sample["id"]], settings), Path(root) / sample["image"])
        record = serialize_inspection(sample, inspection, sample["sha256"], None)
        record["captured_at"] = sample["captured_at"]
        check_acceptance(record)
        records.append(record)
    return records


def score(records):
    return (sum(r['all_matches'] for r in records),
            sum(r['checks']['counts'] for r in records),
            sum(r['checks']['defect_type'] for r in records),
            sum(r['state'] == 'normal' and r['position']['status'] == 'resolved' for r in records))


def tune(root, evaluation, output):
    from src.vision.filtered_detector import DetectorSettings
    root = Path(root).resolve()
    output = fresh_output(root, output)
    evaluation = relative_file(root, evaluation)
    if not evaluation.is_relative_to(root / "outputs"):
        raise ValueError("evaluation must be inside outputs/")
    result = json.loads(evaluation.read_text(encoding="utf-8"))
    raw = json.loads((evaluation.parent / "raw_detections.json").read_text(encoding="utf-8"))
    config, samples = load_samples(root)
    if (result["kind"] != "live_inference" or raw["model_sha256"] != result["model_sha256"]
            or result["model_sha256"] != config["model"]["sha256"]):
        raise ValueError("stored detections must belong to the bundled model")
    raw_by_id = {r["id"]: r for r in raw["records"]}
    if len(raw["records"]) != 100 or set(raw_by_id) != {s["id"] for s in samples}:
        raise ValueError("stored detection identities differ")
    for sample in samples:
        row = raw_by_id[sample["id"]]
        if row["image"] != sample["image"] or row["image_sha256"] != sample["sha256"]:
            raise ValueError("stored detections refer to another image")
    base = DetectorSettings(**raw["settings"])
    best_settings = base
    best_records = replay(root, samples, raw_by_id, base)
    best_score = score(best_records)
    trials = []

    def consider(settings):
        nonlocal best_settings, best_records, best_score
        records = replay(root, samples, raw_by_id, settings)
        value = score(records)
        trials.append({"settings": settings.to_dict(), "score": value})
        if value > best_score:
            best_settings, best_records, best_score = settings, records, value

    thresholds = (.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9)
    for iou in (.3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9):
        for conf in thresholds:
            consider(DetectorSettings(conf=conf, iou=iou, imgsz=base.imgsz,
                                      raw_conf=base.raw_conf, raw_iou=base.raw_iou))
    for _ in range(3):
        before = best_score
        for class_id in range(4):
            current = list(best_settings.class_conf or (best_settings.conf,) * 4)
            for threshold in thresholds:
                candidate = current.copy()
                candidate[class_id] = threshold
                consider(DetectorSettings(conf=best_settings.conf, iou=best_settings.iou, imgsz=base.imgsz,
                                          raw_conf=base.raw_conf, raw_iou=base.raw_iou, class_conf=tuple(candidate)))
        if best_score == before:
            break
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "result.json", {"kind": "stored_detection_replay", "settings": best_settings.to_dict(),
                                           "matches": best_score[0], "records": best_records,
                                           "summary": summarize(best_records),
                                           "notes": "Tuned on these demo images; not independent accuracy."})
    write_new_json(output / "trials.json", trials)
    return best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(tune(ROOT, args.evaluation, args.output))


if __name__ == "__main__":
    main()
