"""2段階の追加学習。既定は設定確認のみ、--runを付けたときだけ学習する。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
import argparse
from collections import Counter
import json
from pathlib import Path

from training.data import (CLASSES, file_hash, fresh_output, prepare_dataset, relative_file,
                           select_records, verify_dataset, write_new_json)

ROOT = Path(__file__).resolve().parents[1]


def training_plan(root, stage, weights, output):
    root = Path(root).resolve()
    records = select_records(verify_dataset(root), stage)
    weights = relative_file(root, weights)
    if weights.suffix != ".pt":
        raise ValueError("existing local .pt weights required; automatic download is disabled")
    output = fresh_output(root, output)
    if weights.is_relative_to(output):
        raise ValueError("input weights overlap output")
    config = json.loads((root / "training/configs" / (stage + ".json")).read_text(encoding="utf-8"))
    settings = {k: v for k, v in config.items()
                if k not in ("model", "data", "project", "name", "save_dir", "task", "mode")}
    settings.update(data=(output / "dataset/dataset.yaml").as_posix(),
                    project=output.as_posix(), name="training", exist_ok=False, resume=False)
    return {"stage": stage, "weights": weights.relative_to(root).as_posix(),
            "weights_sha256": file_hash(weights), "output": output.relative_to(root).as_posix(),
            "counts": dict(Counter(r["split"] for r in records)), "settings": settings}, records


def run_training(root, plan, records, trainer_factory=None):
    root = Path(root).resolve()
    output = fresh_output(root, plan["output"])
    if file_hash(root / plan["weights"]) != plan["weights_sha256"]:
        raise ValueError("input weights changed")
    verify_dataset(root)
    output.mkdir(parents=True, exist_ok=False)
    prepare_dataset(root, records, output / "dataset")
    write_new_json(output / "request.json", plan)
    if trainer_factory is None:
        from ultralytics import YOLO
        trainer_factory = YOLO
    model = trainer_factory(str(root / plan["weights"]))
    model.train(**plan["settings"])
    verify_dataset(root)
    if file_hash(root / plan["weights"]) != plan["weights_sha256"]:
        raise ValueError("input weights changed during training")
    if Path(model.trainer.save_dir).resolve() != output / "training":
        raise ValueError("trainer output directory differs")
    if tuple(model.names[i] for i in range(len(model.names))) != CLASSES:
        raise ValueError("trained class order differs")
    best = output / "training/weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "finetune"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="実際の学習を明示的に開始する")
    args = parser.parse_args()
    plan, records = training_plan(ROOT, args.stage, args.weights, args.output)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.run:
        print(run_training(ROOT, plan, records))
    else:
        print("Check only: no training, download, or output creation.")


if __name__ == "__main__":
    main()
