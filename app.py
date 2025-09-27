from flask import Flask, render_template, request, redirect, url_for
import os
import tempfile
from datetime import datetime, timedelta
import speech_recognition as sr
from textblob import TextBlob
from ollama import chat

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------- Helper Functions ----------
def transcribe_wav(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        return f"[Transcription error]: {e}"

def gemma_feedback(transcript):
    prompt = f"""
You are an AI assistant. Read the transcript below.

Rules:
- Return exactly 3 sections:
  1) Issue
  2) Action
  3) Confirmation and reference

Transcript:
{transcript}
"""
    try:
        resp = chat(
            model="gemma:2b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes customer complaints."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp["message"]["content"]
    except Exception as e:
        return f"[Gemma error]: {e}"

def create_ics_event(summary, description, dt_start: datetime, duration_minutes=60, uid=None):
    dt_end = dt_start + timedelta(minutes=duration_minutes)
    def fmt(dt): return dt.strftime("%Y%m%dT%H%M%S")
    uid = uid or f"{int(dt_start.timestamp())}@aicallint"
    ics = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//AI Call Intelligence//EN\n"
        "BEGIN:VEVENT\n"
        f"UID:{uid}\n"
        f"DTSTAMP:{fmt(datetime.utcnow())}Z\n"
        f"DTSTART:{fmt(dt_start)}\n"
        f"DTEND:{fmt(dt_end)}\n"
        f"SUMMARY:{summary}\n"
        f"DESCRIPTION:{description}\n"
        "END:VEVENT\n"
        "END:VCALENDAR\n"
    )
    return ics

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    transcript_text = ""
    feedback = ""
    sentiment = {}

    if request.method == "POST":
        mode = request.form.get("mode")
        if mode == "Upload WAV":
            file = request.files.get("wav_file")
            if file:
                wav_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(wav_path)
                transcript_text = transcribe_wav(wav_path)
        else:
            transcript_text = request.form.get("manual_transcript", "")

        if transcript_text:
            # Sentiment Analysis
            tb = TextBlob(transcript_text)
            polarity = tb.sentiment.polarity
            if polarity > 0.2:
                sentiment_label = "Positive"
            elif polarity < -0.2:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            sentiment = {"score": polarity, "label": sentiment_label}

            # AI Feedback
            feedback = gemma_feedback(transcript_text)
            return render_template("result.html", transcript=transcript_text, sentiment=sentiment, feedback=feedback)

    return render_template("index.html")

# ---------- Run Flask ----------
if __name__ == "__main__":
    app.run(debug=True)
