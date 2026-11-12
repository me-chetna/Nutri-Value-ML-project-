import unittest
from io import BytesIO
from unittest.mock import patch

from ocr_utils import extract_nutrition, extract_text
from PIL import Image


class ExtractNutritionTests(unittest.TestCase):
    def test_extract_text_reads_uploaded_image_bytes(self):
        image_buffer = BytesIO()
        Image.new("RGB", (4, 4), color="white").save(image_buffer, format="PNG")

        with patch("ocr_utils._get_reader") as get_reader:
            get_reader.return_value.readtext.return_value = [
                "Calories 200",
                "Sodium 400mg",
            ]
            text = extract_text(image_buffer.getvalue())

        self.assertEqual(text, "Calories 200\nSodium 400mg")

    def test_extracts_values_and_normalizes_units(self):
        text = "\n".join([
            "Energy 840 kJ",
            "Calories 202 kcal",
            "Total Fat 8g",
            "Saturated Fat 4g",
            "Total Carbohydrate 26g",
            "Total Sugars 10g",
            "Added Sugars 6g",
            "Protein 4g",
            "Sodium 0.45g",
            "Dietary Fibre 3g",
        ])

        values = extract_nutrition(text)

        self.assertEqual(values["calories"], 202)
        self.assertEqual(values["fat"], 8)
        self.assertEqual(values["carbohydrates"], 26)
        self.assertEqual(values["sugar"], 10)
        self.assertEqual(values["protein"], 4)
        self.assertEqual(values["sodium"], 450)
        self.assertEqual(values["fiber"], 3)

    def test_converts_energy_and_salt_when_only_those_labels_exist(self):
        values = extract_nutrition("Energy 840 kJ\nSalt 1.5g")

        self.assertAlmostEqual(values["calories"], 840 / 4.184)
        self.assertEqual(values["sodium"], 600)

    def test_reads_value_on_line_after_nutrient_name(self):
        values = extract_nutrition("Calories\n200 kcal\nSodium\n400 mg")

        self.assertEqual(values["calories"], 200)
        self.assertEqual(values["sodium"], 400)

    def test_does_not_treat_added_sugar_or_saturated_fat_as_totals(self):
        values = extract_nutrition("Added sugar 5g\nSaturated fat 4g")

        self.assertIsNone(values["sugar"])
        self.assertIsNone(values["fat"])
        self.assertIsNone(values["calories"])


if __name__ == "__main__":
    unittest.main()
