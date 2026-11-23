import unittest

from reweave_decision import (
    PATHWAY_MANUAL,
    PATHWAY_RECOVERY,
    PATHWAY_REPAIR,
    PATHWAY_REUSE,
    evaluate_garment,
)


class TestReWeaveDecisionEngine(unittest.TestCase):
    def test_minor_repair_with_cotton_label_routes_to_repair(self):
        result = evaluate_garment(
            worker_condition="Minor repair needed",
            observed_damage=["Tear"],
            yolo_detections=[{"class_name": "hole", "confidence": 0.72, "area_ratio": 0.02}],
            ocr_result={"composition": "100% cotton", "composition_evidence": "Care-label OCR"},
        )
        self.assertEqual(result["recommended_pathway"], PATHWAY_REPAIR)
        self.assertEqual(result["composition"], "100% cotton")
        self.assertFalse(result["conflict_flag"])
        self.assertTrue(result["human_verification_required"])

    def test_beyond_repair_with_unknown_fibre_requires_material_verification(self):
        result = evaluate_garment(
            worker_condition="Beyond repair",
            observed_damage=["Hole", "Stain"],
            yolo_detections=[{"class_name": "stain", "confidence": 0.85, "area_ratio": 0.15}],
            ocr_result={"composition": "UNKNOWN", "composition_evidence": "None"},
        )
        self.assertEqual(result["recommended_pathway"], PATHWAY_RECOVERY)
        self.assertEqual(result["composition"], "UNKNOWN")
        self.assertTrue(any("UNKNOWN" in r for r in result["reasons"]))

    def test_ready_to_wear_clean_routes_to_reuse(self):
        result = evaluate_garment(
            worker_condition="Ready to wear",
            observed_damage=["No visible damage"],
            yolo_detections=[],
            ocr_result={"composition": "100% cotton"},
        )
        self.assertEqual(result["recommended_pathway"], PATHWAY_REUSE)
        self.assertFalse(result["conflict_flag"])

    def test_conflict_guard_triggers_when_yolo_flags_defect_on_ready_to_wear(self):
        result = evaluate_garment(
            worker_condition="Ready to wear",
            observed_damage=[],
            yolo_detections=[{"class_name": "stain", "confidence": 0.81, "area_ratio": 0.04}],
            ocr_result={"composition": "UNKNOWN"},
        )
        self.assertEqual(result["recommended_pathway"], PATHWAY_MANUAL)
        self.assertTrue(result["conflict_flag"])

    def test_unsure_never_auto_routes_from_yolo_alone(self):
        result = evaluate_garment(
            worker_condition="Unsure — needs inspection",
            yolo_detections=[{"class_name": "hole", "confidence": 0.91, "area_ratio": 0.05}],
        )
        self.assertEqual(result["recommended_pathway"], PATHWAY_MANUAL)
        self.assertTrue(result["human_verification_required"])


if __name__ == "__main__":
    unittest.main()
