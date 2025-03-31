from transformers import pipeline
from textblob import TextBlob
import nltk

nltk.download('punkt')

# Load a sentiment analysis pipeline (can be replaced with a custom model)
analyzer = pipeline("sentiment-analysis")

def detect_manipulation(text):
    # Run sentiment analysis
    sentiment = analyzer(text)[0]

    # Check for exaggeration using TextBlob (high polarity)
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Rule-based detection: if sentiment is extreme & polarity is high
    if sentiment['label'] == "NEGATIVE" and polarity < -0.5:
        return "🚨 Potential Negative Manipulation Detected"
    elif sentiment['label'] == "POSITIVE" and polarity > 0.5:
        return "⚠️ Potential Positive Manipulation Detected"
    else:
        return "✅ No strong manipulation detected"

# Test with sample text
if __name__ == "__main__":
    sample_text = "This is the worst experience ever! I can't believe how terrible this was."
    print("Input Text:", sample_text)
    print(detect_manipulation(sample_text))
