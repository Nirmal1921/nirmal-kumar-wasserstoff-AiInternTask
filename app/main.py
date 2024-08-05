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
        with open(f"data/raw/{image_file.name}", "wb") as f:
            f.write(image_file.getbuffer())
        
        if st.button("Run Pipeline"):
            boxes, labels = segment_image(f"data/raw/{image_file.name}", "data/processed")
            object_images = extract_objects(f"data/raw/{image_file.name}", boxes, "data/processed")
            descriptions = identify_objects(object_images)
            texts = extract_text(object_images)
            summaries = summarize_objects(texts)
            data_map_path = map_data(object_images, descriptions, texts, summaries, "data/metadata")
            generate_output(f"data/raw/{image_file.name}", data_map_path, f"data/processed/{image_file.name}")
            
            st.image(f"data/processed/{image_file.name}", caption="Output Image with Annotations", use_column_width=True)
            st.image(f"data/processed/{image_file.name.replace('.jpg', '_table.png')}", caption="Summary Table", use_column_width=True)

if __name__ == "__main__":
    main()
