# 学習・評価

Windows / Python 3.14.6 / Ultralytics 8.4.138<br>
依存関係はリポジトリ直下の requirements.txt を使用します。

写真80枚・YOLOラベル・学習設定・分割情報を同梱しています。クラス順は bolt / washer / spacer / cap_nut です。
この学習用入口の検証範囲は、データ照合と模擬モデルによる接続テストです。

## 環境

リポジトリ直下で実行します。

```powershell
$appPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'outputs/runtime/yolo'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'outputs/runtime/matplotlib'
```

## 学習

[Ultralytics公式リリース](https://github.com/ultralytics/assets/releases/tag/v8.4.0)の yolo26n.pt を weights/yolo26n.pt に置きます。
SHA-256は [model_provenance.json](model_provenance.json) と照合してください。出力先には新しいフォルダを指定します。

```powershell
& $appPython -m training.train baseline --weights weights/yolo26n.pt --output outputs/baseline --run
& $appPython -m training.train finetune --weights outputs/baseline/training/weights/best.pt --output outputs/finetune --run
```

設定は configs/baseline.json と configs/finetune.json、分割は data/manifest.json にあります。再学習の結果は実行環境により変わります。

## モデルの配布用処理

自分で学習したモデルに対して実行します。重みを保持し、学習用メタデータを除去します。

```powershell
& $appPython -m training.model --input outputs/finetune/training/weights/best.pt --output outputs/distribution/assembly.pt
```

## 評価・設定比較

同梱100画像の数量・OK/NG・欠品位置を正解データと照合します。検出枠はアプリで目視確認します。

```powershell
& $appPython -m training.evaluate --output outputs/demo_evaluation --run
& $appPython -m training.tune --evaluation outputs/demo_evaluation/result.json --output outputs/threshold_comparison
```

学習・評価コマンドは --run を省くと入力確認だけを行います。
同じ100画像を設定調整にも使用しているため、独立した精度評価ではありません。
