import os
from transformers import pipeline

def summarize_objects(texts):
    # Specify the model explicitly to avoid the warning
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    
    summaries = {}
    for image_path, text in texts.items():
        # Skip empty or invalid texts
        if not text.strip():
            summaries[image_path] = "No valid text for summarization."
            continue
        
        word_count = len(text.split())
        
        # Handle very short texts
        if word_count <= 2:
            summaries[image_path] = text  # Directly use the text if it's too short
            continue
        
        # Set min_length and max_length based on word count
        max_length = min(50, word_count)  # Ensure max_length is not greater than word count
        min_length = min(max(5, max_length // 2), max_length - 1)  # Ensure min_length is less than max_length
        
        # Summarize only if there's sufficient text length
        if word_count > 2:
            try:
                summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                summaries[image_path] = summary
            except Exception as e:
                summaries[image_path] = f"Error during summarization: {str(e)}"
        else:
            summaries[image_path] = "Text too short for summarization."
        
    return summaries
