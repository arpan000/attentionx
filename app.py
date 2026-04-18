"""
AttentionX — Main Streamlit App
Upload a long-form video → get viral short-form clips in minutes.

Run with:  streamlit run app.py
"""

import os
import json
import tempfile
import streamlit as st
from pathlib import Path

from transcriber import transcribe, format_for_display
from detector import detect_golden_moments
from exporter import export_all_clips

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttentionX",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ AttentionX")
st.caption("Turn long-form videos into viral short-form clips — automatically.")

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Get a free key at aistudio.google.com",
        value=os.environ.get("GOOGLE_API_KEY", ""),
    )

    whisper_model = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small"],
        index=1,
        help="'base' is recommended — good accuracy, runs on CPU",
    )

    num_clips = st.slider("Number of clips to generate", 1, 10, 5)

    face_detect = st.toggle("Smart face-tracking crop", value=True,
                            help="Uses MediaPipe to keep the speaker centered in vertical frame")

    add_hook = st.toggle("Add hook headline overlay", value=True,
                         help="Adds a punchy 3-second headline at the start of each clip")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Whisper transcribes your video
2. Gemini finds the best moments
3. MoviePy exports vertical clips
4. Captions burned in automatically
""")

# ── Main upload area ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "mkv", "avi", "webm"],
    help="Lectures, podcasts, workshops — any long-form video works",
)

if uploaded_file:
    st.video(uploaded_file)

    if not google_api_key:
        st.warning("Add your Google API key in the sidebar to continue.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = google_api_key

    if st.button("⚡ Generate clips", type="primary", use_container_width=True):

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded file to disk
            video_path = os.path.join(tmpdir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ── Step 1: Transcribe ─────────────────────────────────────────
            with st.status("Transcribing video with Whisper...", expanded=True) as status:
                st.write("Extracting audio...")
                try:
                    transcript = transcribe(
                        video_path,
                        model_size=whisper_model,
                        output_dir=tmpdir,
                    )
                    status.update(
                        label=f"Transcribed! ({len(transcript['segments'])} segments, language: {transcript['language']})",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"Transcription failed: {e}", state="error")
                    st.stop()

            with st.expander("View transcript"):
                st.text(format_for_display(transcript))

            # ── Step 2: Detect golden moments ─────────────────────────────
            with st.status("Finding golden moments with Gemini...", expanded=True) as status:
                try:
                    clips = detect_golden_moments(transcript, n=num_clips)
                    status.update(
                        label=f"Found {len(clips)} clip candidates!",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"Detection failed: {e}", state="error")
                    st.stop()

            st.subheader("Top clip candidates")
            for i, clip in enumerate(clips, 1):
                with st.expander(f"#{i} [{clip['score']}/10] {clip['title']} — {clip['duration']}s"):
                    st.markdown(f"**Hook:** {clip['hook']}")
                    st.markdown(f"**Why it works:** {clip['reason']}")
                    col1, col2 = st.columns(2)
                    col1.metric("Start", f"{int(clip['start_time'] // 60):02d}:{int(clip['start_time'] % 60):02d}")
                    col2.metric("Duration", f"{clip['duration']}s")

            # ── Step 3: Export clips ───────────────────────────────────────
            output_dir = os.path.join(tmpdir, "clips")
            os.makedirs(output_dir, exist_ok=True)

            with st.status("Exporting vertical clips...", expanded=True) as status:
                try:
                    output_paths = export_all_clips(
                        video_path,
                        clips,
                        transcript["segments"],
                        output_dir,
                        add_hook=add_hook,
                        face_detect=face_detect,
                    )
                    status.update(
                        label=f"Exported {len(output_paths)} clips!",
                        state="complete",
                    )
                except Exception as e:
                    status.update(label=f"Export failed: {e}", state="error")
                    st.stop()

            # ── Step 4: Show download buttons ─────────────────────────────
            st.subheader("Download your clips")
            st.caption("Each clip is 9:16 vertical format, ready for TikTok, Reels, or Shorts.")

            for path, clip in zip(output_paths, clips):
                if not os.path.exists(path):
                    continue
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{clip['title']}** — {clip['duration']}s")
                with open(path, "rb") as f:
                    col2.download_button(
                        label="Download",
                        data=f,
                        file_name=Path(path).name,
                        mime="video/mp4",
                        key=f"dl_{clip['title']}",
                    )

            st.success(f"All done! Generated {len(output_paths)} clips ready for posting.")
