import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import torch
import gc
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import cv2
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel

LABELS_MAP = {
    1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane', 6: 'Bus', 7: 'Train', 8: 'Truck',
    9: 'Boat', 10: 'Traffic Light', 11: 'Fire Hydrant', 13: 'Stop Sign', 14: 'Parking Meter', 15: 'Bench',
    16: 'Bird', 17: 'Cat', 18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant', 23: 'Bear', 24: 'Zebra',
    25: 'Giraffe', 27: 'Backpack', 28: 'Umbrella', 31: 'Handbag', 32: 'Tie', 33: 'Suitcase', 34: 'Frisbee',
    35: 'Skis', 36: 'Snowboard', 37: 'Sports Ball', 38: 'Kite', 39: 'Baseball Bat', 40: 'Baseball Glove',
    41: 'Skateboard', 42: 'Surfboard', 43: 'Tennis Racket', 44: 'Bottle', 46: 'Wine Glass', 47: 'Cup', 48: 'Fork',
    49: 'Knife', 50: 'Spoon', 51: 'Bowl', 52: 'Banana', 53: 'Apple', 54: 'Sandwich', 55: 'Orange', 56: 'Broccoli',
    57: 'Carrot', 58: 'Hot Dog', 59: 'Pizza', 60: 'Donut', 61: 'Cake', 62: 'Chair', 63: 'Couch', 64: 'Potted Plant',
    65: 'Bed', 67: 'Dining Table', 70: 'Toilet', 72: 'TV', 73: 'Laptop', 74: 'Mouse', 75: 'Remote', 76: 'Keyboard',
    77: 'Cell Phone', 78: 'Microwave', 79: 'Oven', 80: 'Toaster', 81: 'Sink', 82: 'Refrigerator', 84: 'Book',
    85: 'Clock', 86: 'Vase', 87: 'Scissors', 88: 'Teddy Bear', 89: 'Hair Drier', 90: 'Toothbrush'
}

st.title('AI Pipeline for Image Segmentation and Object Analysis')

def load_model():
    model = maskrcnn_resnet50_fpn(weights="COCO_V1")
    model.eval()
    return model

def preprocess_image(image):
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def get_predictions(model, image):
    with torch.no_grad():
        predictions = model(image)
    return predictions[0]

def extract_text(image):
    text_model = TextExtractionModel()
    return text_model.extract_text(image)

def summarize_text(text):
    summarizer = SummarizationModel()
    return summarizer.summarize(text)

def annotate_image(image, predictions, threshold=0.70):
    img = np.array(image)
    mapped_data = []
    object_id = 1
    
    # Extract and summarize text from the image
    extracted_text = extract_text(image)
    summary = summarize_text(extracted_text) if extracted_text.strip() else 'NA'
    
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            # Convert the box coordinates to integers
            x0, y0, x1, y1 = box.astype(int).tolist()
            object_name = LABELS_MAP.get(int(label), f"Object {label}")
            
            # Draw bounding box
            img = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            
            # Adjust the label font size
            original_font_size = 0.50
            
            img = cv2.putText(img, object_name, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, new_font_size, (255, 0, 0), 1, cv2.LINE_AA)
            
            # Populate table data
            mapped_data.append({
                'Object ID': object_id,
                'Label': object_name,
                'Score': f"{score:.2f}",  # Format score to 2 decimal places
                'Description': f'{object_name} detected',
                'Extracted Text': extracted_text if len(extracted_text.strip()) > 0 else 'NA',
                'Summary': summary
            })
            object_id += 1
    
    # If no objects were detected but text is present, show text and summary only
    if not mapped_data and extracted_text.strip():
        mapped_data.append({
            'Object ID': 'NA',
            'Label': 'NA',
            'Score': 'NA',
            'Description': 'Text only image',
            'Extracted Text': extracted_text,
            'Summary': summary
        })
    
    return Image.fromarray(img), mapped_data

# Adjust the display settings for Streamlit's table
def display_table(mapped_data):
    # Convert mapped_data to DataFrame
    df = pd.DataFrame(mapped_data)
    
    # Display the table with dynamic sizing
    st.write("Mapped Data:")
    st.dataframe(df.style.set_properties(**{
        'text-align': 'left',
        'font-size': '12px',
        'padding': '5px',
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('font-size', '14px')]},
        {'selector': 'td', 'props': [('max-width', '300px'), ('white-space', 'pre-wrap')]},
    ]))

# File upload
image_file = st.file_uploader('Upload an image', type=['jpg', 'png'])

if image_file:
    # Load and preprocess the image
    image = Image.open(image_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    model = load_model()

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Get predictions
    predictions = get_predictions(model, preprocessed_image)

    # Post-process predictions
    scores = predictions['scores'].numpy()
    labels = predictions['labels'].numpy()
    boxes = predictions['boxes'].detach().cpu().numpy()

    st.write(f"Detected objects count: {len(labels)}")

    if len(labels) > 0 or extract_text(image).strip():
        annotated_image, mapped_data = annotate_image(image, {'boxes': boxes, 'labels': labels, 'scores': scores})
        
        # Display annotated image
        st.image(annotated_image, caption='Processed Image with Object Labels', use_column_width=True)

        # Display the mapped data table dynamically
        display_table(mapped_data)
    else:
        st.write("No objects detected and no text found. Please try another image.")

    # Cleanup resources
    del preprocessed_image, predictions, annotated_image, image_file, model
    torch.cuda.empty_cache()
    gc.collect()
