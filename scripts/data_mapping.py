import os
import json

def map_data(object_images, descriptions, texts, summaries, output_dir):
    data_map = {}
    
    for image_path in object_images:
        data_map[image_path] = {
            "description": descriptions.get(image_path, []),
            "text": texts.get(image_path, ""),
            "summary": summaries.get(image_path, "")
        }
        
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_map.json")
    with open(output_path, 'w') as f:
        json.dump(data_map, f)
        
    return output_path
