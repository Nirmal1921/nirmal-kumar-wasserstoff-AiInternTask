import unittest
from scripts.text_extraction import extract_objects
from PIL import Image

class TestExtraction(unittest.TestCase):
    def test_extract_objects(self):
        image = Image.open("data/raw/sample_image.jpg")
        segments = [{"mask": None, "bbox": [10, 20, 30, 40]}]  # Mocked segments
        extracted_objects = extract_objects(image, segments)
        self.assertTrue(len(extracted_objects) > 0, "Object extraction failed, no objects extracted")

if __name__ == "__main__":
    unittest.main()
