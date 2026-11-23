import unittest
from PIL import Image

from reweave_ocr import (
    extract_care_instructions,
    parse_care_label,
    preprocess_label_image,
    run_label_ocr,
)


class _DummyReader:
    """Mock EasyOCR reader for fast, deterministic unit tests."""

    def __init__(self, responses):
        self._responses = list(responses)

    def readtext(self, image_array, detail=1, paragraph=False):
        if self._responses:
            return self._responses.pop(0)
        return []


class TestReWeaveOCR(unittest.TestCase):
    def test_preprocess_upscales_small_label_and_preserves_rgb(self):
        small_label = Image.new("RGB", (200, 100), color=(240, 240, 240))
        out = preprocess_label_image(small_label)
        self.assertEqual(out.shape, (200, 400, 3))  # 2x upscale, 3 RGB channels

    def test_extracts_care_instructions_accurately(self):
        text = "100% Cotton\nMachine wash cold. Do not bleach. Do not tumble dry. Warm iron."
        instructions = extract_care_instructions(text)
        self.assertIn("Machine wash cold", instructions)
        self.assertIn("Do not bleach", instructions)
        self.assertIn("Do not tumble dry", instructions)
        self.assertIn("Low / warm iron", instructions)
        self.assertNotIn("Machine wash", instructions)  # No duplicate generic entry

    def test_parse_care_label_marks_evidence_when_composition_found(self):
        parsed = parse_care_label("COTTON: 80% POLYESTER: 20%\nHand wash only", 0.91)
        self.assertEqual(parsed["composition"], "80% cotton + 20% polyester")
        self.assertIn("Care-label OCR", parsed["composition_evidence"])
        self.assertEqual(parsed["care_instructions"], ["Hand wash"])

    def test_parse_care_label_keeps_unknown_when_unreadable(self):
        parsed = parse_care_label("", None)
        self.assertEqual(parsed["composition"], "UNKNOWN")
        self.assertEqual(parsed["care_instructions"], [])
        self.assertIn("None", parsed["composition_evidence"])

    def test_run_label_ocr_falls_back_to_raw_if_preprocessed_misses_composition(self):
        img = Image.new("RGB", (300, 200), color=(255, 255, 255))
        reader = _DummyReader(
            [
                [(None, "Blurry text", 0.40)],          # 1st call (preprocessed)
                [(None, "100% Cotton", 0.92)],          # 2nd call (raw fallback)
            ]
        )
        result = run_label_ocr(img, reader)
        self.assertEqual(result["composition"], "100% cotton")
        self.assertFalse(result["preprocessing_used"])


if __name__ == "__main__":
    unittest.main()
