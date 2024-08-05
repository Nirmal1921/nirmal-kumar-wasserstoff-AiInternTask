import os
from PIL import Image

def extract_objects(image_path, boxes, output_dir):
    image = Image.open(image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    object_images = []
    for i, box in enumerate(boxes):
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        output_path = os.path.join(output_dir, f'object_{i}.png')
        cropped_image.save(output_path)
        object_images.append(output_path)
        
    return object_images
