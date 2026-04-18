# ⚡ AttentionX — Automated Content Repurposing Engine

> Turn a single 60-minute mentorship session into a week's worth of viral short-form content — automatically.

**Built for the UnsaidTalks × AttentionX AI Hackathon**

---

## Demo Video

🎥 [Watch the demo](YOUR_GOOGLE_DRIVE_LINK_HERE)

---

## What it does

AttentionX takes a long-form video (lecture, podcast, workshop) and:

1. **Transcribes** it using OpenAI Whisper with accurate word-level timestamps
2. **Finds golden moments** using Google Gemini 1.5 Flash — high-energy, quotable, viral-worthy segments
3. **Exports vertical clips** (9:16) with MediaPipe face-tracking so the speaker stays centered
4. **Burns in karaoke captions** and a punchy hook headline automatically

All in one click through a clean Streamlit web interface.

---

## Tech stack

| Layer | Tool |
|---|---|
| Transcription | OpenAI Whisper (`base` model) |
| AI analysis | Google Gemini 1.5 Flash |
| Video editing | MoviePy |
| Face tracking | MediaPipe |
| UI + hosting | Streamlit |

---

## Project structure

```
attentionx/
├── app.py            # Streamlit UI — entry point
├── transcriber.py    # Module 1: Whisper transcription
├── detector.py       # Module 2: Gemini golden moment detection
├── exporter.py       # Module 3: Vertical crop + captions export
├── requirements.txt
└── README.md
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/attentionx.git
cd attentionx
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Whisper also requires `ffmpeg`. Install it with:
> - Mac: `brew install ffmpeg`
> - Ubuntu: `sudo apt install ffmpeg`
> - Windows: download from [ffmpeg.org](https://ffmpeg.org)

### 3. Set your API key

```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How to use

1. Enter your Google API key in the sidebar (or set it as an environment variable)
2. Upload a long-form video (MP4, MOV, MKV, etc.)
3. Click **Generate clips**
4. Wait ~2–5 minutes depending on video length
5. Download your vertical clips and post them!

---

## Module breakdown

### `transcriber.py`
- Extracts audio from video using MoviePy
- Runs Whisper to produce timestamped transcript segments
- Saves `transcript.json` for downstream use

### `detector.py`
- Formats transcript for Gemini with timestamps
- Prompts Gemini to identify top N clip moments
- Returns clip candidates with title, hook headline, and viral score

### `exporter.py`
- Cuts each clip from the source video
- Uses MediaPipe face detection to center the vertical crop
- Burns karaoke captions from transcript segments
- Adds hook headline overlay for the first 3 seconds
- Exports as 1080×1920 MP4

---

## Evaluation criteria alignment

| Criterion | How AttentionX addresses it |
|---|---|
| Impact (20%) | End-to-end pipeline — upload to downloadable clips in minutes |
| Innovation (20%) | Gemini for moment detection + MediaPipe smart crop combined |
| Technical execution (20%) | Modular, well-documented Python code |
| User experience (25%) | Clean Streamlit UI, no technical setup needed for end user |
| Presentation (15%) | See demo video link above |

---

*Built with ❤️ for the AttentionX AI Hackathon by UnsaidTalks*
