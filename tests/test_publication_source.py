"""公開用ソースの入力保護と分割・評価を検証。実モデルは使用しない。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from training.data import (CLASSES, STATE_COUNTS, file_hash, fresh_output, label_counts,
                           prepare_dataset, relative_file, select_records, verify_dataset)
from training.metrics import check_acceptance, summarize
from training.train import run_training, training_plan


class PublicationSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.records = []
        for direction, size, train_size in (("front", 10, 8), ("right", 3, 2), ("left", 3, 2)):
            for state, counts in STATE_COUNTS.items():
                for n in range(1, size + 1):
                    sample_id = f"{direction}_{state}_{n:03d}"
                    row = {"id": sample_id, "direction": direction, "state": state,
                           "split": "train" if n <= train_size else "val"}
                    text = "".join(f"{i} 0.5 0.5 0.1 0.1\n" for i, count in enumerate(counts) for _ in range(count))
                    for key, name, value in (("image", f"data/images/{sample_id}.jpg", sample_id),
                                             ("label", f"data/labels/{sample_id}.txt", text)):
                        path = self.root / "training" / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(value, encoding="utf-8")
                        row[key], row[key + "_sha256"] = name, file_hash(path)
                    self.records.append(row)
        self.save_manifest()
        config_dir = self.root / "training/configs"
        config_dir.mkdir()
        for stage in ("baseline", "finetune"):
            (config_dir / (stage + ".json")).write_text(json.dumps({"epochs": 50, "device": "cpu", "seed": 42}), encoding="utf-8")
        (self.root / "weights.pt").write_bytes(b"TEST ONLY; not a model")

    def save_manifest(self):
        (self.root / "training/data/manifest.json").write_text(json.dumps({"records": self.records}), encoding="utf-8")

    def test_both_splits_and_plan_do_not_create_outputs(self):
        for stage, counts in (("baseline", {"train": 40, "val": 10}), ("finetune", {"train": 60, "val": 20})):
            plan, selected = training_plan(self.root, stage, "weights.pt", "outputs/" + stage)
            self.assertEqual(plan["counts"], counts)
            self.assertEqual(len(selected), sum(counts.values()))
            self.assertEqual(plan["settings"]["epochs"], 50)
        self.assertFalse((self.root / "outputs").exists())

    def test_changed_image_is_rejected(self):
        (self.root / "training" / self.records[0]["image"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed training"):
            verify_dataset(self.root)

    def test_rehashed_wrong_class_counts_are_rejected(self):
        path = self.root / "training" / self.records[0]["label"]
        path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        self.records[0]["label_sha256"] = file_hash(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "class counts"):
            verify_dataset(self.root)

    def test_split_change_is_rejected(self):
        self.records[0]["split"] = "val"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "split or category"):
            verify_dataset(self.root)

    def test_extra_training_file_is_rejected(self):
        (self.root / "training/data/images/extra.jpg").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra or missing"):
            verify_dataset(self.root)

    def test_path_escape_and_missing_weights_are_rejected(self):
        for path in ("../outside.pt", "C:/outside.pt"):
            with self.assertRaises(ValueError):
                relative_file(self.root, path)
        with self.assertRaises(ValueError):
            training_plan(self.root, "baseline", "missing.pt", "outputs/baseline")

    def test_existing_output_is_preserved(self):
        output = self.root / "outputs/keep"
        output.mkdir(parents=True)
        sentinel = output / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            fresh_output(self.root, "outputs/keep")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_training_connection_uses_selected_dataset_and_local_weights(self):
        plan, selected = training_plan(self.root, "baseline", "weights.pt", "outputs/baseline")
        calls = []
        class FakeTrainer:
            names = dict(enumerate(CLASSES))
            def __init__(inner, weights):
                calls.append(weights)
            def train(inner, **settings):
                calls.append(settings)
                saved = Path(settings["project"]) / settings["name"]
                (saved / "weights").mkdir(parents=True)
                (saved / "weights/best.pt").write_bytes(b"FAKE OUTPUT")
                inner.trainer = SimpleNamespace(save_dir=saved)
        best = run_training(self.root, plan, selected, trainer_factory=FakeTrainer)
        self.assertEqual(best.read_bytes(), b"FAKE OUTPUT")
        self.assertEqual(calls[0], str(self.root / "weights.pt"))
        dataset = self.root / "outputs/baseline/dataset"
        self.assertEqual(len(list((dataset / "images/train").glob("*.jpg"))), 40)
        self.assertEqual(len(list((dataset / "images/val").glob("*.jpg"))), 10)
        self.assertEqual(json.loads((dataset / "dataset.yaml").read_text())["names"], list(CLASSES))
        self.assertEqual(len(verify_dataset(self.root)), 80)

    def test_nonfinite_labels_are_rejected(self):
        path = self.root / "bad.txt"
        path.write_text("0 nan 0.5 0.1 0.1", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid normalized"):
            label_counts(path)

    def test_wrong_side_is_not_counted_as_correct(self):
        record = {"id": "test", "direction": "front", "state": "washer_front_missing",
                  "counts": dict(zip(CLASSES, (1, 1, 1, 1))), "inspection_result": "NG",
                  "defect_type": "washer_after_spacer_missing",
                  "position": {"status": "resolved", "missing_defect_type": "washer_after_spacer_missing"}}
        check_acceptance(record)
        self.assertFalse(record["all_matches"])
        summary = summarize([record])["overall"]
        self.assertEqual(summary["quantity_matches"], 1)
        self.assertEqual(summary["wrong_side"], 1)
        self.assertEqual(summary["side_correct"], 0)


if __name__ == "__main__":
    unittest.main()
