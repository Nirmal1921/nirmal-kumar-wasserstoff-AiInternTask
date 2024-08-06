import unittest
from main_file import annotate_image
from PIL import Image

class TestIdentification(unittest.TestCase):

    def setUp(self):
        self.image = Image.open('tests/sample_image.jpg')  # Replace with a valid test image path
        self.predictions = {
            'boxes': [[50, 50, 100, 100]],
            'labels': [1],
            'scores': [0.9]
        }

    def test_annotation_output(self):
        annotated_image, mapped_data = annotate_image(self.image, self.predictions)
        self.assertIsInstance(mapped_data, list, "Mapped data should be a list")
        self.assertIsInstance(annotated_image, Image.Image, "Annotated image should be a PIL Image object")
        self.assertGreater(len(mapped_data), 0, "Mapped data should contain entries")

    def test_mapped_data_structure(self):
        _, mapped_data = annotate_image(self.image, self.predictions)
        self.assertIn('Object ID', mapped_data[0], "Mapped data should contain 'Object ID'")
        self.assertIn('Label', mapped_data[0], "Mapped data should contain 'Label'")
        self.assertIn('Score', mapped_data[0], "Mapped data should contain 'Score'")

if __name__ == '__main__':
    unittest.main()
