import torch
from yolov5 import YOLOv5
from PIL import Image

def identify_objects(object_images):
    model = YOLOv5('yolov5s.pt', device='cpu')
    
    descriptions = {}
    for image_path in object_images:
        image = Image.open(image_path)
        results = model.predict(image)  # Correct usage to get predictions
        descriptions[image_path] = results.pandas().xyxy[0].to_dict(orient="records")
        
    return descriptions
