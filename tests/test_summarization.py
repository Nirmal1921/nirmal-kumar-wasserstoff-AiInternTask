import unittest
from main_file import summarize_text

class TestSummarization(unittest.TestCase):

    def test_summarize_text(self):
        sample_text = "This is a sample text that needs to be summarized. It contains important information that should be retained."
        summary = summarize_text(sample_text)
        self.assertIsInstance(summary, str, "Summary should be a string")
        self.assertLess(len(summary), len(sample_text), "Summary should be shorter than the original text")

if __name__ == '__main__':
    unittest.main()
