import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from reweave_core import confirm_inspection, fetch_inspections, init_db, save_inspection


class TestReWeaveDBMigration(unittest.TestCase):
    def test_migrates_legacy_db_without_losing_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_db = Path(tmpdir) / "legacy.db"
            # Create an old-schema table exactly like the original MVP
            with sqlite3.connect(legacy_db) as conn:
                conn.execute(
                    """
                    CREATE TABLE inspections (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        image_name TEXT NOT NULL,
                        garment_type TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        observed_damage TEXT NOT NULL,
                        care_label_text TEXT NOT NULL,
                        composition TEXT NOT NULL,
                        recommendation TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        worker_notes TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO inspections VALUES (
                        '318A96DE', '2026-11-23T10:00:00+00:00', 'old_shirt.jpg',
                        'Shirt', 'Ready to wear', '[]', '', 'UNKNOWN',
                        'REUSE', 'Old record', ''
                    )
                    """
                )
                conn.commit()

            # Run new init_db() migration
            init_db(db_path=legacy_db)
            df = fetch_inspections(db_path=legacy_db)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["id"], "318A96DE")
            self.assertEqual(df.iloc[0]["confirmation_status"], "PENDING")
            self.assertEqual(json.loads(df.iloc[0]["yolo_detections"]), [])

    def test_saves_evidence_and_worker_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "evidence.db"
            insp_id = save_inspection(
                {
                    "image_name": "stained_tee.jpg",
                    "garment_type": "T-shirt",
                    "condition": "Minor repair needed",
                    "observed_damage": ["Stain"],
                    "care_label_text": "100% Cotton\nMachine wash cold",
                    "composition": "100% cotton",
                    "composition_evidence": "Care-label OCR",
                    "care_instructions": ["Machine wash cold"],
                    "ocr_confidence": 0.91,
                    "yolo_detections": [{"class_name": "stain", "confidence": 0.86}],
                    "recommendation": "REPAIR",
                    "rationale": "Minor repairable damage.",
                },
                db_path=test_db,
            )

            df_before = fetch_inspections(db_path=test_db)
            row_before = df_before.iloc[0]
            self.assertEqual(row_before["confirmation_status"], "PENDING")
            self.assertAlmostEqual(float(row_before["yolo_max_confidence"]), 0.86)
            self.assertEqual(json.loads(row_before["care_instructions"]), ["Machine wash cold"])

            updated = confirm_inspection(
                insp_id,
                confirmed_pathway="REPAIR",
                worker_notes="Confirmed small washable stain",
                db_path=test_db,
            )
            self.assertTrue(updated)

            df_after = fetch_inspections(db_path=test_db)
            row_after = df_after.iloc[0]
            self.assertEqual(row_after["confirmation_status"], "CONFIRMED")
            self.assertEqual(row_after["confirmed_pathway"], "REPAIR")
            self.assertIsNotNone(row_after["confirmed_at"])


if __name__ == "__main__":
    unittest.main()
