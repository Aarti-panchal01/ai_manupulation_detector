# 🧠 AI Manipulation Detector  
*Detect persuasive or manipulative intent in speech and text using AI.*

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build](https://img.shields.io/badge/status-prototype-lightgrey)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

---

## 🔍 Overview

**AI Manipulation Detector** is an AI-powered tool that takes **voice input**, transcribes it using **OpenAI Whisper**, and analyzes the resulting text to detect **persuasive**, **manipulative**, or **emotionally biased** language.

It is designed to raise awareness of subtle psychological techniques used in conversations — from marketing to everyday interactions.

---

## ✨ Features

- 🎤 **Speech-to-text transcription** (OpenAI Whisper)
- 💬 **Text classification** for manipulation techniques
- 📊 **Scoring system** indicating persuasion intensity
- 🧠 Uses **RoBERTa** + custom-trained models
- 🌐 **API-powered backend** (FastAPI)
- 🖥️ Simple frontend for interactive testing

---

## 💡 Use Cases

- Personal awareness in conversations
- Educational tool for communication analysis
- Ethical AI research
- Real-time transcription + sentiment inspection

---

## ⚙️ Tech Stack

| Component        | Tool / Framework            |
|------------------|-----------------------------|
| Speech-to-Text   | OpenAI Whisper              |
| NLP Model        | Hugging Face Transformers (RoBERTa) |
| Backend API      | FastAPI                     |
| Frontend         | HTML / JavaScript           |
| Deployment Ready | ✅ (Locally or on Cloud)    |

---

## 🚀 Getting Started

## 1. Clone the Repository
git clone https://github.com/Aarti-panchal01/ai_manupulation_detector.git
cd ai_manupulation_detector

## 2. Set Up Python Environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

## 3. Run the App
uvicorn main:app --reload
---

## 🙋‍♀️ Author
Made with 💡 and curiosity by Aarti Panchal
GitHub: @Aarti-panchal01

