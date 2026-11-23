import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from reweave_core import (
    extract_composition,
    fetch_inspections,
    image_diagnostics,
    init_db,
    prepare_image,
    recommend_pathway,
    save_inspection,
)


class TestReWeaveCore(unittest.TestCase):
    # --- 1. Care-label composition extraction tests ---
    def test_unknown_when_no_explicit_composition(self):
        self.assertEqual(extract_composition("Wash cold. Do not bleach."), "UNKNOWN")

    def test_unknown_when_label_text_is_empty(self):
        self.assertEqual(extract_composition(""), "UNKNOWN")

    def test_extracts_explicit_composition(self):
        self.assertEqual(
            extract_composition("60% Cotton 40% Polyester"),
            "60% cotton + 40% polyester",
        )

    def test_extracts_fibre_before_percentage(self):
        self.assertEqual(
            extract_composition("COTTON: 60% POLYESTER - 40%"),
            "60% cotton + 40% polyester",
        )

    def test_deduplicates_repeated_ocr_lines(self):
        self.assertEqual(
            extract_composition("100% Cotton\nCotton 100%"),
            "100% cotton",
        )

    def test_rejects_zero_and_impossible_percentage(self):
        self.assertEqual(extract_composition("0% Cotton"), "UNKNOWN")
        self.assertEqual(extract_composition("150% Cotton"), "UNKNOWN")

    # --- 2. Pathway routing tests (all 4 worker conditions) ---
    def test_routes_ready_to_wear_to_reuse(self):
        self.assertEqual(recommend_pathway("Ready to wear")[0], "REUSE")

    def test_routes_minor_repair_to_repair(self):
        self.assertEqual(recommend_pathway("Minor repair needed")[0], "REPAIR")

    def test_routes_beyond_repair_to_testing(self):
        self.assertEqual(
            recommend_pathway("Beyond repair")[0],
            "MATERIAL TESTING / RECYCLING ASSESSMENT",
        )

    def test_routes_uncertain_to_manual_inspection(self):
        self.assertEqual(
            recommend_pathway("Unsure — needs inspection")[0],
            "MANUAL INSPECTION",
        )

    def test_unknown_condition_defaults_to_manual_inspection(self):
        self.assertEqual(
            recommend_pathway("Unexpected value")[0],
            "MANUAL INSPECTION",
        )

    # --- 3. Image preprocessing, diagnostics & invalid upload tests ---
    def test_prepare_image_converts_rgba_png_to_rgb(self):
        rgba_img = Image.new("RGBA", (120, 80), color=(255, 0, 0, 128))
        prepared = prepare_image(rgba_img)
        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.size, (120, 80))

    def test_image_diagnostics_returns_expected_metrics(self):
        sample_img = Image.new("RGB", (200, 100), color=(100, 100, 100))
        diag = image_diagnostics(sample_img)
        self.assertEqual(diag["width"], 200)
        self.assertEqual(diag["height"], 100)
        self.assertEqual(diag["mean_brightness"], 100.0)
        self.assertEqual(diag["sharpness_proxy"], 0.0)

    def test_invalid_image_bytes_raises_unidentified_image_error(self):
        corrupted_upload = BytesIO(b"this is definitely not an image file")
        with self.assertRaises(UnidentifiedImageError):
            Image.open(corrupted_upload)

    # --- 4. SQLite persistence & dashboard count aggregation tests ---
    def test_save_inspection_handles_optional_fields_safely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test_minimal.db"
            init_db(db_path=test_db)
            insp_id = save_inspection(
                {
                    "image_name": "minimal.jpg",
                    "garment_type": "Shirt",
                    "condition": "Ready to wear",
                    "recommendation": "REUSE",
                },
                db_path=test_db,
            )
            df = fetch_inspections(db_path=test_db)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["id"], insp_id)
            self.assertEqual(df.iloc[0]["composition"], "UNKNOWN")
            self.assertEqual(json.loads(df.iloc[0]["observed_damage"]), [])

    def test_sqlite_multiple_inspections_persistence_and_dashboard_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test_reweave.db"
            init_db(db_path=test_db)

            sample_records = [
                ("shirt.jpg", "Shirt", "Ready to wear", ["No visible damage"], "100% Cotton"),
                ("jeans.jpg", "Jeans", "Minor repair needed", ["Tear"], "Cotton: 98% Elastane: 2%"),
                ("rag.jpg", "Fabric scrap", "Beyond repair", ["Hole", "Stain"], ""),
                ("kurta.jpg", "Other", "Unsure — needs inspection", ["Other / unsure"], "Wash cold"),
            ]

            saved_ids = []
            for img_name, g_type, cond, damage, label_txt in sample_records:
                rec_pathway, rationale = recommend_pathway(cond)
                comp = extract_composition(label_txt)
                insp_id = save_inspection(
                    {
                        "image_name": img_name,
                        "garment_type": g_type,
                        "condition": cond,
                        "observed_damage": damage,
                        "care_label_text": label_txt,
                        "composition": comp,
                        "recommendation": rec_pathway,
                        "rationale": rationale,
                        "worker_notes": "Phase 1 automated test",
                    },
                    db_path=test_db,
                )
                saved_ids.append(insp_id)

            df = fetch_inspections(db_path=test_db)

            self.assertEqual(len(df), 4)
            self.assertEqual(set(df["id"]), set(saved_ids))

            self.assertEqual(int((df["recommendation"] == "REUSE").sum()), 1)
            self.assertEqual(int((df["recommendation"] == "REPAIR").sum()), 1)
            self.assertEqual(
                int((df["recommendation"] == "MATERIAL TESTING / RECYCLING ASSESSMENT").sum()),
                1,
            )
            self.assertEqual(int((df["recommendation"] == "MANUAL INSPECTION").sum()), 1)

            rag_row = df[df["image_name"] == "rag.jpg"].iloc[0]
            self.assertEqual(rag_row["composition"], "UNKNOWN")
            self.assertEqual(json.loads(rag_row["observed_damage"]), ["Hole", "Stain"])


if __name__ == "__main__":
    unittest.main()
