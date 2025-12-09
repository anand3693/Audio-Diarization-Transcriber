# 🎙️ Speaker-Aware Audio Transcription (Whisper + Pyannote)

This project performs **speech transcription** and **speaker diarization** using state-of-the-art AI models:

- **Faster-Whisper** → High-accuracy speech-to-text
- **Pyannote** → Speaker identification and segmentation

The output includes **timestamps**, **speaker labels**, and **transcribed dialogue** for every spoken chunk.

---

## 🚀 Features
✔ Transcribes speech using Whisper (Large-V3)  
✔ Identifies multiple speakers using Pyannote diarization  
✔ Supports GPU (CUDA) acceleration  
✔ Outputs speaker-tagged subtitles with time ranges  
✔ Uses `.env` for secure Hugging Face token management  

---

## 📂 Project Structure
.
├── main.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md

## 1. Create & Activate Virtual Environment

python -m venv venv
venv\Scripts\activate  # Windows
### or
source venv/bin/activate  # Linux/Mac

## 2. Install Dependencies

pip install -r requirements.txt

## 3. Add Hugging Face Token

Create a .env file in root:

HF_TOKEN=hf_your_token_here

## 4. Update Audio File Path in Code

Inside main.py, set your MP3/WAV file path:

AUDIO_FILE = r"Path\to\your\audio.mp3"

## 5. Run the Application

python main.py
