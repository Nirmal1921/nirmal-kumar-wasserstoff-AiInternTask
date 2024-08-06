from transformers import pipeline

class SummarizationModel:
    def __init__(self, model_name='sshleifer/distilbart-cnn-12-6'):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str) -> str:
        """
        Summarize the given text using a pre-trained model from Hugging Face.
        """
        if not text.strip():
            return "No valid text for summarization."
        
        word_count = len(text.split())
        
        # Adjust the summarization parameters based on the length of the text
        if word_count <= 2:
            return text
        
        max_length = min(50, word_count)
        min_length = min(max(5, max_length // 2), max_length - 1)
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary.strip()  # Strip any extraneous whitespace from the summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"
