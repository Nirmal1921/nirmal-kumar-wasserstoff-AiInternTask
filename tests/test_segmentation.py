import unittest
import torch
from PIL import Image
from main_file import load_model, preprocess_image, get_predictions

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.model = load_model()
        self.image = Image.open('tests/sample_image.jpg')  # Replace with a valid test image path
        self.preprocessed_image = preprocess_image(self.image)

    def test_model_output_type(self):
        predictions = get_predictions(self.model, self.preprocessed_image)
        self.assertIsInstance(predictions, dict, "Model output should be a dictionary")

    def test_boxes_shape(self):
        predictions = get_predictions(self.model, self.preprocessed_image)
        self.assertGreater(len(predictions['boxes']), 0, "Predicted boxes should not be empty")

    def test_model_inference(self):
        predictions = get_predictions(self.model, self.preprocessed_image)
        scores = predictions['scores'].numpy()
        self.assertTrue((scores >= 0).all() and (scores <= 1).all(), "Scores should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()
