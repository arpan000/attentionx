"""
AttentionX — Module 3: Clip Exporter
Takes clip candidates and exports them as vertical (9:16) short-form videos
with burned-in karaoke-style captions.

Requires: moviepy, mediapipe, Pillow
"""

import os
import json
import tempfile
import textwrap
from pathlib import Path

import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont


# ── Face tracking (MediaPipe) ─────────────────────────────────────────────────

def detect_face_center(frame_rgb: np.ndarray) -> tuple[float, float] | None:
    """
    Run MediaPipe face detection on a single RGB frame.
    Returns (cx, cy) as fractions of frame width/height, or None if no face found.
    """
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        results = detector.process(frame_rgb)
        if not results.detections:
            return None
        det = results.detections[0]
        box = det.location_data.relative_bounding_box
        cx = box.xmin + box.width / 2
        cy = box.ymin + box.height / 2
        return cx, cy


def get_face_x_positions(clip: VideoFileClip, sample_every: float = 1.0) -> list[float]:
    """
    Sample frames from the clip every `sample_every` seconds, run face detection,
    and collect the horizontal face center positions (as pixel x coords).
    Returns a list of detected x positions.
    """
    positions = []
    t = 0.0
    while t < clip.duration:
        frame = clip.get_frame(t)  # numpy array, RGB
        result = detect_face_center(frame)
        if result:
            cx, _ = result
            positions.append(cx * clip.w)  # convert fraction to pixel x
        t += sample_every
    return positions


# ── Vertical crop ─────────────────────────────────────────────────────────────

def crop_to_vertical(clip: VideoFileClip, face_positions: list[float]) -> VideoFileClip:
    """
    Crop a 16:9 clip to 9:16 (vertical), keeping the speaker in frame.
    Target resolution: 1080 × 1920 (standard Reels/TikTok).

    Strategy:
      - Compute the target crop width: clip.h * (9/16)
      - If face positions available, center the crop window on the median face x
      - Clamp to valid pixel range
      - Resize height to 1920, width to 1080
    """
    target_w = int(clip.h * 9 / 16)  # width of 9:16 crop in original pixels
    target_h = clip.h

    if face_positions:
        # Median face x position avoids outliers from wrong detections
        median_x = float(np.median(face_positions))
        # Center crop window on face, clamped to frame boundaries
        x1 = median_x - target_w / 2
        x1 = max(0, min(x1, clip.w - target_w))
    else:
        # Fallback: center crop
        x1 = (clip.w - target_w) / 2

    x1 = int(x1)

    cropped = clip.crop(x1=x1, y1=0, x2=x1 + target_w, y2=target_h)
    resized = cropped.resize(height=1920)  # moviepy maintains aspect ratio

    print(f"[Exporter] Cropped to vertical: face-centered at x={x1}px, output 1080×1920")
    return resized


# ── Captions ──────────────────────────────────────────────────────────────────

def get_segments_for_clip(
    all_segments: list[dict],
    start_time: float,
    end_time: float,
) -> list[dict]:
    """
    Filter transcript segments that fall within [start_time, end_time].
    Adjusts timestamps to be relative to the clip start.
    """
    clip_segments = []
    for seg in all_segments:
        # Include segment if it overlaps with the clip window
        if seg["end"] <= start_time or seg["start"] >= end_time:
            continue
        clip_segments.append({
            "start": max(0.0, seg["start"] - start_time),
            "end":   min(end_time - start_time, seg["end"] - start_time),
            "text":  seg["text"],
        })
    return clip_segments


def make_caption_clip(
    text: str,
    duration: float,
    video_w: int,
    video_h: int,
    fontsize: int = 52,
) -> "CompositeVideoClip":
    """
    Create a TextClip for a single caption segment.
    Style: bold white text with black outline — high contrast for all backgrounds.
    Positioned in the lower third of the frame.
    """
    # Wrap long lines
    wrapped = "\n".join(textwrap.wrap(text, width=22))

    caption = (
        TextClip(
            wrapped,
            fontsize=fontsize,
            font="DejaVu-Sans-Bold",
            color="white",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            size=(video_w - 80, None),  # leave 40px margin each side
            align="center",
        )
        .set_duration(duration)
        .set_position(("center", video_h * 0.72))  # lower-third
    )
    return caption


def add_captions_to_clip(
    video_clip: VideoFileClip,
    caption_segments: list[dict],
) -> CompositeVideoClip:
    """
    Overlay all caption segments onto the video clip as a composite.
    Each segment appears and disappears at its exact timestamp (karaoke style).
    """
    caption_clips = []
    for seg in caption_segments:
        duration = seg["end"] - seg["start"]
        if duration <= 0 or not seg["text"].strip():
            continue
        cap = (
            make_caption_clip(seg["text"], duration, video_clip.w, video_clip.h)
            .set_start(seg["start"])
        )
        caption_clips.append(cap)

    if not caption_clips:
        return video_clip  # no captions, return unchanged

    return CompositeVideoClip([video_clip, *caption_clips])


# ── Hook headline overlay ─────────────────────────────────────────────────────

def add_hook_headline(
    video_clip: VideoFileClip,
    hook_text: str,
    display_duration: float = 3.0,
) -> CompositeVideoClip:
    """
    Add a 'hook' headline at the very top of the clip for the first few seconds.
    Bold, centered, upper region — grabs attention before the viewer scrolls past.
    """
    wrapped = "\n".join(textwrap.wrap(hook_text, width=26))
    headline = (
        TextClip(
            wrapped,
            fontsize=44,
            font="DejaVu-Sans-Bold",
            color="#FFDD44",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            size=(video_clip.w - 60, None),
            align="center",
        )
        .set_duration(min(display_duration, video_clip.duration))
        .set_start(0)
        .set_position(("center", video_clip.h * 0.08))  # top region
    )
    return CompositeVideoClip([video_clip, headline])


# ── Main export function ───────────────────────────────────────────────────────

def export_clip(
    source_video_path: str,
    clip_info: dict,
    all_segments: list[dict],
    output_dir: str,
    add_hook: bool = True,
    face_detect: bool = True,
    fps: int = 30,
) -> str:
    """
    Export a single short-form clip from the source video.

    Args:
        source_video_path: Path to the original long-form video
        clip_info:         A clip dict from detector.detect_golden_moments()
        all_segments:      All transcript segments from transcriber.transcribe()
        output_dir:        Where to save the exported .mp4
        add_hook:          Whether to overlay the hook headline at the start
        face_detect:       Whether to use MediaPipe for smart vertical crop
        fps:               Output frame rate

    Returns:
        Path to the exported .mp4 file
    """
    start = clip_info["start_time"]
    end   = clip_info["end_time"]
    title = clip_info["title"]
    hook  = clip_info.get("hook", "")

    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:40]
    output_path = os.path.join(output_dir, f"{safe_title}.mp4")

    print(f"\n[Exporter] Processing clip: '{title}' ({start}s → {end}s)")

    with VideoFileClip(source_video_path) as source:
        # 1. Cut the clip
        raw_clip = source.subclip(start, end)
        print(f"[Exporter]   Cut: {raw_clip.duration:.1f}s at {raw_clip.w}×{raw_clip.h}")

        # 2. Smart vertical crop
        if face_detect:
            print("[Exporter]   Running face detection (sampling every 1s)...")
            face_xs = get_face_x_positions(raw_clip, sample_every=1.0)
            print(f"[Exporter]   Face detected in {len(face_xs)} frames")
        else:
            face_xs = []

        vertical_clip = crop_to_vertical(raw_clip, face_xs)

        # 3. Get caption segments for this clip
        caption_segs = get_segments_for_clip(all_segments, start, end)
        print(f"[Exporter]   Adding {len(caption_segs)} caption segments")

        # 4. Add karaoke captions
        captioned = add_captions_to_clip(vertical_clip, caption_segs)

        # 5. Add hook headline
        if add_hook and hook:
            captioned = add_hook_headline(captioned, hook)
            print(f"[Exporter]   Hook overlay: '{hook}'")

        # 6. Export
        print(f"[Exporter]   Exporting to: {output_path}")
        captioned.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

    print(f"[Exporter]   Done: {output_path}")
    return output_path


def export_all_clips(
    source_video_path: str,
    clips: list[dict],
    all_segments: list[dict],
    output_dir: str,
    **kwargs,
) -> list[str]:
    """
    Export all clip candidates. Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for i, clip in enumerate(clips, 1):
        print(f"\n── Clip {i}/{len(clips)} ──")
        try:
            path = export_clip(source_video_path, clip, all_segments, output_dir, **kwargs)
            output_paths.append(path)
        except Exception as e:
            print(f"[Exporter] ERROR on clip '{clip['title']}': {e}")
    return output_paths


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from transcriber import load_transcript

    if len(sys.argv) < 4:
        print("Usage: python exporter.py <video.mp4> <transcript.json> <clips_manifest.json>")
        sys.exit(1)

    video_path = sys.argv[1]
    transcript = load_transcript(sys.argv[2])
    with open(sys.argv[3]) as f:
        clips = json.load(f)

    output_dir = "output_clips"
    paths = export_all_clips(
        video_path,
        clips[:3],  # export top 3 for testing
        transcript["segments"],
        output_dir,
    )
    print(f"\n── Exported {len(paths)} clips to '{output_dir}/' ──")
    for p in paths:
        print(f"  {p}")
