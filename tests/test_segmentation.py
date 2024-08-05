import unittest
from scripts.segmentation import segment_image
from PIL import Image

class TestSegmentation(unittest.TestCase):
    def test_segment_image(self):
        image = Image.open("data/raw/sample_image.jpg")
        segments = segment_image(image)
        self.assertTrue(len(segments) > 0, "Segmentation failed, no segments found")

if __name__ == "__main__":
    unittest.main()
