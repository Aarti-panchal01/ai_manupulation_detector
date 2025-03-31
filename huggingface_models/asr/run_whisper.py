import whisper

# Load the Whisper model
model = whisper.load_model("base")

# Path to the audio file
audio_path = "audio_files/sample_audio.mp3"

# Transcribe the audio
result = model.transcribe(audio_path)

# Print the transcribed text
print("Transcription:", result["text"])
