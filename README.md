# ASSEMBLY INSIGHT

部品の組付け写真から、部品の数量とワッシャの欠品位置を判定するPythonアプリです。検査結果をSQLiteに保存し、品質傾向の集計とGemma 4 26Bによる分析コメントの生成を行います。

## 起動

Windows / Python 3.14.6<br>
Ollamaをインストールして起動してください。リポジトリ直下で実行します。

```powershell
python -m venv .venv
$appPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $appPython -m pip install -r requirements.txt
ollama pull gemma4:26b
& $appPython -m streamlit run local_app.py --server.address 127.0.0.1
```

「検査ワークスペース」で画像を一括検査し、「品質ダッシュボード」で分析します。部品検出モデルとサンプル画像100枚を同梱しています。

## 資料

[設計](ARCHITECTURE.md) / [モデル・データ](MODEL_CARD.md) / [学習・評価手順](training/README.md)

## ライセンス

Copyright (C) 2026 岡田 耕（Okada Kou）<br>
コード・文書・学習済みモデル：[AGPL-3.0](LICENSE)<br>
写真・ラベル・正解データ：[CC0](ASSET_LICENSE.md)<br>
[第三者ソフトウェアの表記](THIRD_PARTY_NOTICES.md)
