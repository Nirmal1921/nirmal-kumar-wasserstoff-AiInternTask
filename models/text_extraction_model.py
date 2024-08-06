import pytesseract
from PIL import Image

class TextExtractionModel:
    def extract_text(self, image):
        # Convert the image to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        # Run text extraction
        text = pytesseract.image_to_string(image)
        return text
