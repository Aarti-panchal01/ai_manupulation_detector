import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
import speech_recognition as sr
import transformers
from textblob import TextBlob
import librosa
import langdetect
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load transformer model for persuasion detection
try:
    model_name = "facebook/bart-large-mnli"
    classifier = transformers.pipeline("zero-shot-classification", model=model_name)
    logger.info("Transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load transformer model: {str(e)}")
    classifier = None

# Expanded persuasion categories
persuasion_labels = [
    "emotional appeal", "guilt-tripping", "fear-mongering", "logical reasoning",
    "trust-building", "manipulation", "authority appeal", "scarcity tactic",
    "social proof", "reciprocity"
]

# Analyze persuasion techniques
def analyze_persuasion(text):
    if not classifier:
        return [("Error", "Model not loaded")]
    try:
        result = classifier(text, persuasion_labels, multi_label=True)
        detected_techniques = [(label, score) for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
        return detected_techniques if detected_techniques else [("None", 0.0)]
    except Exception as e:
        logger.error(f"Persuasion analysis failed: {str(e)}")
        return [("Error", str(e))]

# Sentiment analysis
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0.1:
            return "Positive"
        elif sentiment_score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return "Unknown"

# Language detection
def detect_language(text):
    try:
        return langdetect.detect(text).upper()
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return "Unknown"

# Speaker detection (basic segmentation using librosa)
def detect_speakers(file_path):
    try:
        y, sr = librosa.load(file_path)
        segments = librosa.effects.split(y, top_db=20)
        return len(segments) if len(segments) > 1 else 1
    except Exception as e:
        logger.error(f"Speaker detection failed: {str(e)}")
        return 0

# Generate PDF report
def generate_pdf_report(data, output_path):
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "AI Audio Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(100, 710, f"Transcribed Text: {data['text'][:100]}...")
        c.drawString(100, 690, f"Sentiment: {data['sentiment']}")
        c.drawString(100, 670, f"Language: {data['language']}")
        c.drawString(100, 650, f"Speakers Detected: {data['speakers']}")
        c.drawString(100, 630, "Persuasion Techniques:")
        y = 610
        for tech, score in data['persuasion_techniques']:
            c.drawString(120, y, f"{tech}: {(score * 100):.1f}%")
            y -= 20
        c.save()
        logger.info(f"PDF report generated at {output_path}")
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    logger.info("Received request to /analyze endpoint")
    
    if "audio" not in request.files:
        logger.warning("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        logger.warning(f"Unsupported file format: {file.filename}")
        return jsonify({"error": "Unsupported file format"}), 400

    file_path = os.path.join("uploads", file.filename)
    try:
        # Save file synchronously
        file.save(file_path)
        logger.info(f"File saved: {file_path}")

        # Convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            logger.info("Attempting speech-to-text conversion")
            text = recognizer.recognize_google(audio_data)
        logger.info("Speech-to-text conversion successful")

        # Analyze audio
        sentiment = analyze_sentiment(text)
        persuasion_analysis = analyze_persuasion(text)
        language = detect_language(text)
        speakers = detect_speakers(file_path)

        # Prepare response
        response = {
            "text": text,
            "sentiment": sentiment,
            "language": language,
            "speakers": speakers,
            "persuasion_techniques": persuasion_analysis
        }

        # Generate PDF
        pdf_path = os.path.join("uploads", "report.pdf")
        generate_pdf_report(response, pdf_path)

        logger.info("Analysis completed successfully")
        return jsonify(response)

    except sr.UnknownValueError as e:
        logger.error(f"Speech recognition failed: Could not understand audio - {str(e)}")
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        logger.error(f"Speech recognition request failed: {str(e)}")
        return jsonify({"error": f"Speech recognition service error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File removed: {file_path}")

@app.route("/download_report")
def download_report():
    pdf_path = os.path.join("uploads", "report.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name="audio_analysis_report.pdf")
    else:
        return jsonify({"error": "Report not found"}), 404

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)

