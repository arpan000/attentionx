"""
AttentionX — Module 1: Transcription
Uses OpenAI Whisper to convert video audio into a timestamped transcript.
Output is a list of segments: [{start, end, text}, ...] used by all other modules.
"""

import os
import whisper
import tempfile
import json
from pathlib import Path


def extract_audio(video_path: str, output_dir: str = None) -> str:
    """
    Extract audio from a video file as a .wav file.
    Uses moviepy — installed as a shared dependency across all modules.
    Returns path to the extracted audio file.
    """
    from moviepy.editor import VideoFileClip

    video_path = Path(video_path)
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    audio_path = os.path.join(output_dir, f"{video_path.stem}_audio.wav")

    print(f"[Transcriber] Extracting audio from: {video_path.name}")
    clip = VideoFileClip(str(video_path))
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()
    print(f"[Transcriber] Audio saved to: {audio_path}")

    return audio_path


def transcribe(
    video_path: str,
    model_size: str = "base",
    language: str = None,
    output_dir: str = None,
) -> dict:
    """
    Transcribe a video file using OpenAI Whisper.

    Args:
        video_path:  Path to the input video (.mp4, .mov, .mkv, etc.)
        model_size:  Whisper model to use. Options:
                       "tiny"   — fastest, least accurate (~1GB VRAM)
                       "base"   — good balance for hackathon use ✓ recommended
                       "small"  — better accuracy, slower
                       "medium" — high accuracy, needs decent GPU
        language:    Force a language code (e.g. "en"). None = auto-detect.
        output_dir:  Where to save transcript JSON. Defaults to a temp folder.

    Returns:
        A dict with:
          - "text":     Full plain transcript as a single string
          - "segments": List of {id, start, end, text} dicts (word-level timestamps)
          - "language": Detected or forced language code
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Step 1: Extract audio
    audio_path = extract_audio(video_path, output_dir)

    # Step 2: Load Whisper model
    print(f"[Transcriber] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    # Step 3: Transcribe
    print("[Transcriber] Transcribing audio... (this may take a moment)")
    options = {"verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    # Step 4: Clean up and structure output
    segments = [
        {
            "id": seg["id"],
            "start": round(seg["start"], 2),   # seconds
            "end":   round(seg["end"],   2),   # seconds
            "text":  seg["text"].strip(),
        }
        for seg in result["segments"]
    ]

    transcript = {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language", "unknown"),
    }

    # Step 5: Save transcript JSON alongside audio
    transcript_path = os.path.join(output_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"[Transcriber] Done. {len(segments)} segments transcribed.")
    print(f"[Transcriber] Language detected: {transcript['language']}")
    print(f"[Transcriber] Transcript saved to: {transcript_path}")

    # Clean up temporary audio file
    os.remove(audio_path)

    return transcript


def load_transcript(transcript_path: str) -> dict:
    """
    Load a previously saved transcript JSON from disk.
    Useful for re-running later pipeline stages without re-transcribing.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_for_display(transcript: dict, max_chars: int = 80) -> str:
    """
    Format transcript segments into a readable, timestamped string.
    Useful for debugging or displaying in Streamlit.

    Example output:
        [00:01 → 00:08]  Welcome everyone, today we're going to talk about...
        [00:08 → 00:15]  The most important thing about leadership is...
    """
    lines = []
    for seg in transcript["segments"]:
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        text  = seg["text"]
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        lines.append(f"[{start} → {end}]  {text}")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Convert float seconds to MM:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <path_to_video>")
        sys.exit(1)

    video = sys.argv[1]
    result = transcribe(video, model_size="base")

    print("\n── Transcript preview ──")
    print(format_for_display(result))
