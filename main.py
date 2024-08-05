import sys
import os
# Add the scripts directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

import streamlit as st
from scripts.segmentation import segment_image
from scripts.object_extraction import extract_objects
from scripts.object_identification import identify_objects
from scripts.text_extraction import extract_text
from scripts.attribute_summary import summarize_objects
from scripts.data_mapping import map_data
from scripts.output_generation import generate_output

def main():
    st.title("AI Image Segmentation and Analysis Pipeline")
    
    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        
        # Save the uploaded image
        raw_image_path = f"data/raw/{image_file.name}"
        os.makedirs(os.path.dirname(raw_image_path), exist_ok=True)
        with open(raw_image_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        if st.button("Run Pipeline"):
            try:
                # Segment the image
                boxes, labels = segment_image(raw_image_path, "data/processed")
                
                # Extract objects
                object_images = extract_objects(raw_image_path, boxes, "data/processed")
                
                # Identify objects
                descriptions = identify_objects(object_images)
                
                # Extract text
                texts = extract_text(object_images)
                
                # Summarize objects
                summaries = summarize_objects(texts)
                
                # Map data
                data_map_path = map_data(object_images, descriptions, texts, summaries, "data/metadata")
                
                # Generate output
                output_image_path = f"data/processed/{image_file.name}"
                generate_output(raw_image_path, data_map_path, output_image_path)
                
                # Display results
                st.image(output_image_path, caption="Output Image with Annotations", use_column_width=True)
                
                summary_table_path = output_image_path.replace('.jpg', '_table.png')
                if os.path.exists(summary_table_path):
                    st.image(summary_table_path, caption="Summary Table", use_column_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()