"""信頼済みの学習済み重みから、新しい推論用ファイルを作る。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
import argparse
from pathlib import Path
import pickletools
import re
import zipfile
from training.data import relative_file


def clean_model(source, target):
    """信頼済みの重みを読み、推論不要の学習履歴を除く。"""
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint["model"]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    args = {k: v for k, v in checkpoint.get("train_args", {}).items()
            if k in ("task", "imgsz", "single_cls")}
    model.args = args.copy()
    model.criterion = None
    cleaned = {k: checkpoint[k] for k in ("version", "license", "docs") if k in checkpoint}
    cleaned.update(model=model, train_args=args)
    torch.save(cleaned, target)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    actual = loaded["model"].state_dict()
    if state.keys() != actual.keys() or any(
        state[k].dtype != actual[k].dtype or not torch.equal(state[k], actual[k]) for k in state
    ):
        raise ValueError("学習済みパラメータが変化しました")
    if loaded["model"].names != model.names or loaded.get("license") != checkpoint.get("license"):
        raise ValueError("クラス名またはライセンス表記が変化しました")
    with zipfile.ZipFile(target) as archive:
        payload = archive.read(next(n for n in archive.namelist() if n.endswith("/data.pkl")))
    strings = [arg for _, arg, _ in pickletools.genops(payload) if isinstance(arg, str)]
    if any(re.search(r"(?:^|[\s'\"])[A-Za-z]:[\\/]|/(?:home|Users)/|WindowsPath", s) for s in strings):
        raise ValueError("モデルに内部パスまたは学習情報が残っています")
    return {"kind": "inference_checkpoint", "state_tensors_equal": len(state),
            "class_names_equal": True, "license_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = relative_file(root, args.input)
    target = (root / args.output).resolve()
    if not target.is_relative_to(root / 'outputs') or target.suffix != '.pt' or target.exists():
        raise ValueError('new .pt output below outputs/ required')
    target.parent.mkdir(parents=True, exist_ok=True)
    print(clean_model(source, target))


if __name__ == '__main__':
    main()
