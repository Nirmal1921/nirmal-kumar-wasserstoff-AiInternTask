import unittest
from main_file import extract_text
from PIL import Image

class TestTextExtraction(unittest.TestCase):

    def setUp(self):
        self.image = Image.open('tests/sample_image_with_text.jpg')  # Replace with a valid test image path

    def test_extract_text(self):
        extracted_text = extract_text(self.image)
        self.assertIsInstance(extracted_text, str, "Extracted text should be a string")
        self.assertGreater(len(extracted_text), 0, "Extracted text should not be empty")

if __name__ == '__main__':
    unittest.main()
