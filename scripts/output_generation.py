import matplotlib.pyplot as plt
import cv2
import json

def generate_output(image_path, data_map_path, output_path):
    image = cv2.imread(image_path)
    with open(data_map_path, 'r') as f:
        data_map = json.load(f)
    
    for i, (obj_path, data) in enumerate(data_map.items()):
        obj_image = cv2.imread(obj_path)
        cv2.putText(image, f"Obj {i+1}: {data['summary']}", (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite(output_path, image)
    
    # Generate table
    rows = [["Object", "Description", "Extracted Text", "Summary"]]
    for i, (obj_path, data) in enumerate(data_map.items()):
        rows.append([f"Object {i+1}", data["description"], data["text"], data["summary"]])
    
    plt.table(cellText=rows, colLabels=["Object", "Description", "Extracted Text", "Summary"], loc='center')
    plt.axis('off')
    plt.savefig(output_path.replace(".jpg", "_table.png"))
