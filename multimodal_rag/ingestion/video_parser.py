# ============================================================
#  ingestion/video_parser.py — Video ingestion
#  Audio → Whisper transcription
#  Frames → CLIP image description (optional)
# ============================================================

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from config import WHISPER_MODEL, VIDEO_FRAME_INTERVAL

logger = logging.getLogger(__name__)


def parse_video(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a video file:
      1. Transcribe audio using Whisper (timestamped segments)
      2. Sample key frames and generate descriptions (if CLIP available)

    Returns:
        List of dicts: { text, metadata: { source, type, timestamp, segment } }
    """
    file_name = Path(file_path).name
    logger.info(f"[VIDEO] Parsing: {file_name}")

    chunks = []

    # ── Step 1: Transcribe audio ──
    transcript_chunks = _transcribe_audio(file_path, file_name)
    chunks.extend(transcript_chunks)

    # ── Step 2: Sample frames + describe ──
    frame_chunks = _extract_frame_descriptions(file_path, file_name)
    chunks.extend(frame_chunks)

    logger.info(f"[VIDEO] Total chunks from {file_name}: {len(chunks)}")
    return chunks


def _transcribe_audio(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """Transcribe using Whisper. Returns one chunk per ~30s segment."""
    try:
        import whisper
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    logger.info(f"[VIDEO] Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)

    logger.info(f"[VIDEO] Transcribing audio...")
    result = model.transcribe(file_path, verbose=False)

    chunks = []
    # Group segments into ~30s windows
    buffer_text = []
    buffer_start = 0.0
    buffer_duration = 0.0

    for seg in result.get("segments", []):
        buffer_text.append(seg["text"].strip())
        if buffer_duration == 0:
            buffer_start = seg["start"]
        buffer_duration += (seg["end"] - seg["start"])

        if buffer_duration >= 30.0:
            combined = " ".join(buffer_text)
            if combined.strip():
                chunks.append({
                    "text": combined,
                    "metadata": {
                        "source":    file_name,
                        "file_path": file_path,
                        "type":      "video_transcript",
                        "timestamp": _format_time(buffer_start),
                        "segment":   f"{_format_time(buffer_start)} — {_format_time(seg['end'])}",
                    }
                })
            buffer_text = []
            buffer_start = seg["end"]
            buffer_duration = 0.0

    # Flush remaining
    if buffer_text:
        combined = " ".join(buffer_text)
        if combined.strip():
            chunks.append({
                "text": combined,
                "metadata": {
                    "source":    file_name,
                    "file_path": file_path,
                    "type":      "video_transcript",
                    "timestamp": _format_time(buffer_start),
                    "segment":   f"{_format_time(buffer_start)} — end",
                }
            })

    logger.info(f"[VIDEO] Transcribed {len(chunks)} segments from {file_name}")
    return chunks


def _extract_frame_descriptions(file_path: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Sample key frames from video and generate text descriptions.
    Uses pytesseract for any on-screen text (slides, subtitles).
    CLIP-based semantic description is optional.
    """
    chunks = []
    try:
        import cv2
        from PIL import Image
        import numpy as np

        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % VIDEO_FRAME_INTERVAL == 0:
                timestamp_sec = frame_count / fps
                # Try OCR on frame (good for lecture slides)
                ocr_text = _ocr_frame(frame)
                if ocr_text and len(ocr_text) > 20:
                    chunks.append({
                        "text": f"[VIDEO FRAME at {_format_time(timestamp_sec)}]\n{ocr_text}",
                        "metadata": {
                            "source":    file_name,
                            "file_path": file_path,
                            "type":      "video_frame",
                            "timestamp": _format_time(timestamp_sec),
                            "frame_num": frame_count,
                        }
                    })

            frame_count += 1

        cap.release()
        logger.info(f"[VIDEO] Extracted {len(chunks)} frame text chunks from {file_name}")

    except Exception as e:
        logger.warning(f"[VIDEO] Frame extraction failed (optional): {e}")

    return chunks


def _ocr_frame(frame) -> str:
    """Run pytesseract OCR on a video frame."""
    try:
        import pytesseract
        import cv2
        from PIL import Image
        import numpy as np

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        text = pytesseract.image_to_string(pil_img).strip()
        return text
    except Exception:
        return ""


def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
