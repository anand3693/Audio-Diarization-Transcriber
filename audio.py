import os
from dotenv import load_dotenv
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

AUDIO_FILE = r"D:\Audio\LearningEnglishConversations-20251111-TheEnglishWeSpeakNoTwoWaysAboutIt.mp3"


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    use_auth_token=HF_TOKEN
)
diarization = pipeline(AUDIO_FILE)


model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")

segments, _ = model.transcribe(AUDIO_FILE, beam_size=5, language=None)

results = []
for seg in segments:
    text = seg.text
    start, end = seg.start, seg.end

    speaker = "Unknown"
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        if start >= turn.start and end <= turn.end:
            speaker = speaker_label
            break

    results.append({
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text
    })

for r in results:
    print(f"[{r['speaker']} - {r['start']:.2f}s]: {r['text']}")
