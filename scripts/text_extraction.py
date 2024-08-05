import pytesseract
from PIL import Image

# Ensure the path to the tesseract executable is correctly set if it's not in your PATH
# For example, on Windows you might need to add this line
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(object_images):
    texts = {}
    for image_path in object_images:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        texts[image_path] = text.strip()
        
    return texts
