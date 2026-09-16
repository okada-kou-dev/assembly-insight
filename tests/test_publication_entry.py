import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.asset_catalog import load_catalog
from src.analysis.comment_summary import build_quality_summary
from src.llm.client import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "local_app.py" if (ROOT / "local_app.py").is_file() else ROOT / "publication/local_app.py"


class TestPublicationEntry(unittest.TestCase):
    def test_bundled_settings_and_numeric_first_generation_keep_existing_contract(self):
        catalog = load_catalog()
        sample = catalog.samples[0]
        records = [{"image_id": 1, "image_path": str(sample.path), "captured_at": "2026-09-01T01:15:00",
                    "inspection_result": "NG", "defect_type": "spacer_missing", "confidence": .9}]
        events = []

        def generate(context, **kwargs):
            self.assertIn("chart", events)
            self.assertEqual(kwargs["model"], DEFAULT_MODEL)
            self.assertEqual(context["quality_summary"], build_quality_summary(records, data_type="portfolio_demo"))
            events.append("llm")
            return "画面接続テスト用の模擬本文。"

        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            db = Path(directory) / "fixture.db"
            db.write_bytes(b"Fixture; database access is mocked")
            with patch("src.ui.inspection_panel.render_batch_inspection") as panel, \
                 patch("src.storage.database.list_inspections", return_value=records) as read_db, \
                 patch("src.ui.quality_dashboard._chart", side_effect=lambda chart: events.append("chart")), \
                 patch("src.llm.client.generate_free_quality_comment", side_effect=generate) as llm:
                app = AppTest.from_file(str(ENTRY), default_timeout=20).run()
                self.assertEqual(len(app.exception), 0)
                self.assertIsNone(app.session_state["analysis_result"])
                self.assertEqual(panel.call_args.kwargs["model_path"], catalog.model_path)
                self.assertEqual(panel.call_args.kwargs["detector_settings"], catalog.settings)
                llm.assert_not_called()
                app.text_input(key="quality_db_path").set_value(str(db)).run()
                app.button(key="public_quality_run").click().run()
                read_db.assert_not_called()
                llm.assert_not_called()
                self.assertEqual(len(app.metric), 0)
                self.assertTrue(any("先に「検査ワークスペース」" in item.value for item in app.info))
                # 検査パネルを模擬しているため、保存完了時のセッション状態を与える。
                app.session_state["completed_inspection_db"] = str(db.resolve())
                app.button(key="public_quality_run").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(events[-1], "llm")
                self.assertEqual([metric.value for metric in app.metric], ["1", "0", "1", "100.0%"])
                app.run()
                llm.assert_called_once()
                read_db.assert_called_once()
                app.text_input(key="quality_db_path").set_value(str(db.parent / "other.db")).run()
                app.button(key="public_quality_run").click().run()
                self.assertIsNone(app.session_state["analysis_result"])
                self.assertEqual(len(app.metric), 0)
                llm.assert_called_once()
                read_db.assert_called_once()


    def test_catalog_requires_bundled_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"schema_version": 1, "synthetic_datetime_labels": True,
                      "samples": [{"id": "outside", "path": "../outside.png", "view": "front", "sha256": ""}],
                      "settings": {}, "model": {"path": "../outside.pt", "sha256": ""}}
            (root / "catalog.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_catalog(root)


if __name__ == "__main__":
    unittest.main()
