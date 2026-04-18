"""
AttentionX — Module 2: Golden Moment Detector
Sends the full transcript to Google Gemini and asks it to identify
the highest-impact segments worth turning into short-form clips.

Input:  transcript dict from transcriber.py
Output: list of clip candidates [{start, end, title, hook, reason, score}, ...]
"""

import os
import json
import google.generativeai as genai
from transcriber import format_for_display


# Configure Gemini — set your API key in the environment:
#   export GOOGLE_API_KEY="your-key-here"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SYSTEM_PROMPT = """
You are an expert social media content strategist and video editor.
Your job is to analyse transcripts of long-form educational videos (lectures,
podcasts, mentorship sessions) and identify the best short-form clip moments.

A great clip has ONE or more of these qualities:
  - A surprising insight or counterintuitive idea
  - A concrete story or vivid example
  - A clear, quotable takeaway ("the one thing you need to know is...")
  - An emotional high point — passion, humour, vulnerability
  - A strong opening hook (question, bold claim, or relatable problem)

Clips should be 30–90 seconds long (prefer 45–75 seconds as the sweet spot).
Avoid clips that start or end mid-sentence. Pick natural speech boundaries.

Return ONLY valid JSON — no markdown, no explanation, no backticks.
"""

EXTRACT_PROMPT = """
Here is a timestamped transcript. Each line is [MM:SS → MM:SS] followed by text.

{transcript}

Identify the top {n} clip moments. For each, return a JSON array where every
element has exactly these fields:
  - "start_time"  : float, clip start in seconds
  - "end_time"    : float, clip end in seconds
  - "title"       : string, a 3–6 word internal clip title (e.g. "Why failure builds resilience")
  - "hook"        : string, a punchy 1-sentence hook headline for TikTok/Reels (max 12 words)
  - "reason"      : string, 1–2 sentences explaining why this clip will perform well
  - "score"       : integer 1–10, your confidence this will go viral

Return the array sorted by score descending. Output JSON only.
"""


def detect_golden_moments(
    transcript: dict,
    n: int = 5,
    model_name: str = "gemini-1.5-flash",
) -> list[dict]:
    """
    Use Gemini to detect the best clip moments in a transcript.

    Args:
        transcript:  Output dict from transcriber.transcribe()
        n:           How many clip candidates to return (default 5)
        model_name:  Gemini model to use. "gemini-1.5-flash" is fast and free-tier friendly.

    Returns:
        List of clip candidate dicts, sorted by score (highest first).
    """
    print(f"[Detector] Sending transcript to Gemini ({model_name})...")

    formatted = format_for_display(transcript, max_chars=120)
    prompt = EXTRACT_PROMPT.format(transcript=formatted, n=n)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Gemini sometimes wraps JSON in backticks despite instructions — strip them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        clips = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Detector] Failed to parse Gemini response as JSON: {e}")
        print(f"[Detector] Raw response:\n{raw[:500]}")
        raise

    # Validate and clamp timestamps to transcript bounds
    total_duration = transcript["segments"][-1]["end"] if transcript["segments"] else 0
    validated = []
    for clip in clips:
        start = max(0.0, float(clip.get("start_time", 0)))
        end   = min(total_duration, float(clip.get("end_time", start + 60)))
        duration = end - start

        if duration < 10:
            print(f"[Detector] Skipping clip '{clip.get('title')}' — too short ({duration:.0f}s)")
            continue

        validated.append({
            "start_time": start,
            "end_time":   end,
            "duration":   round(duration, 1),
            "title":      clip.get("title", "Untitled clip"),
            "hook":       clip.get("hook", ""),
            "reason":     clip.get("reason", ""),
            "score":      int(clip.get("score", 5)),
        })

    validated.sort(key=lambda x: x["score"], reverse=True)

    print(f"[Detector] Found {len(validated)} valid clip candidates.")
    for i, clip in enumerate(validated, 1):
        print(f"  {i}. [{clip['score']}/10] {clip['title']} ({clip['duration']}s)")

    return validated


def save_clips_manifest(clips: list[dict], output_dir: str) -> str:
    """Save the clip candidates to a JSON manifest file."""
    path = os.path.join(output_dir, "clips_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)
    print(f"[Detector] Manifest saved to: {path}")
    return path


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from transcriber import load_transcript

    if len(sys.argv) < 2:
        print("Usage: python detector.py <path_to_transcript.json>")
        sys.exit(1)

    transcript = load_transcript(sys.argv[1])
    clips = detect_golden_moments(transcript, n=5)

    print("\n── Top clips ──")
    for clip in clips:
        print(f"\n[{clip['score']}/10] {clip['title']}")
        print(f"  Time:   {clip['start_time']}s → {clip['end_time']}s ({clip['duration']}s)")
        print(f"  Hook:   {clip['hook']}")
        print(f"  Why:    {clip['reason']}")
