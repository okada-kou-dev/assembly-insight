# 設計

画像検出 → 数量・位置判定 → SQLite保存 → 品質集計 → グラフ・分析コメントの順に処理します。
集計値はPythonで計算し、LLMには8つのCSVとして渡します。

| 処理 | 実装 |
| --- | --- |
| 起動・資産読込み | `local_app.py`、`src/asset_catalog.py` |
| 部品検出 | `src/vision/` |
| 数量・ワッシャ欠品位置の判定 | `inspection_rules.py`、`src/inspection/` |
| 検査履歴の保存 | `src/storage/` |
| 日別・6時間帯別集計、統計計算 | `src/analysis/` |
| Gemma 4との通信 | `src/llm/` |
| 画面・グラフ | `src/ui/` |

既存DBの上書き時はバックアップを作成します。
LLMとの通信に失敗しても集計結果は保持します。

## テスト

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```
