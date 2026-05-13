"""
video_processing.py
-------------------
Handles all video I/O operations:
- Video validation (format, duration ≤ 40 s)
- Frame extraction using OpenCV
- 15-second segment splitting
- Thumbnail generation
"""

import cv2
import numpy as np
from PIL import Image
import logging
import os
from typing import List, Dict, Tuple, Optional

# ─── Logger ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_VIDEO_DURATION_SEC: float = 40.0      # Hard upload limit
SEGMENT_DURATION_SEC: float   = 15.0     # Each analysis window
FRAMES_PER_SEGMENT: int        = 16      # Frames sampled per 15 s segment
SUPPORTED_FORMATS: List[str]   = [".mp4", ".mov", ".avi"]
THUMBNAIL_SIZE: Tuple[int, int] = (320, 180)


# ─── Classes ─────────────────────────────────────────────────────────────────
class VideoProcessingError(Exception):
    """Raised when video cannot be read or is invalid."""


class VideoProcessor:
    """
    Handles loading, validating, and segmenting a sports video clip.

    Usage
    -----
    vp = VideoProcessor("/tmp/match.mp4")
    info = vp.get_info()
    segments = vp.extract_segments()
    """

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise VideoProcessingError(f"File not found: {video_path}")
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise VideoProcessingError(
                f"Unsupported format '{ext}'. Allowed: {SUPPORTED_FORMATS}"
            )
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration: float = 0.0
        self._load_metadata()

    # ── Private helpers ──────────────────────────────────────────────────────
    def _load_metadata(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError(
                f"OpenCV could not open video: {self.video_path}"
            )
        self._fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration     = self._total_frames / self._fps
        cap.release()
        logger.info(
            "Video loaded: %.1f s  |  %.0f fps  |  %dx%d",
            self._duration, self._fps, self._width, self._height,
        )

    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> Optional[Image.Image]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ── Public API ────────────────────────────────────────────────────────────
    def get_info(self) -> Dict:
        """Return basic video metadata as a dict."""
        return {
            "path":         self.video_path,
            "duration_sec": round(self._duration, 2),
            "fps":          round(self._fps, 2),
            "width":        self._width,
            "height":       self._height,
            "total_frames": self._total_frames,
        }

    def validate_duration(self) -> None:
        """Raise VideoProcessingError if clip exceeds MAX_VIDEO_DURATION_SEC."""
        if self._duration > MAX_VIDEO_DURATION_SEC:
            raise VideoProcessingError(
                f"Video is {self._duration:.1f} s long. "
                f"Maximum allowed duration is {MAX_VIDEO_DURATION_SEC:.0f} s."
            )

    def get_thumbnail(self, at_second: float = 0.5) -> Image.Image:
        """Return a single thumbnail frame resized to THUMBNAIL_SIZE."""
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = int(at_second * self._fps)
        img = self._read_frame(cap, frame_idx)
        cap.release()
        if img is None:
            # blank fallback
            img = Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 30))
        return img.resize(THUMBNAIL_SIZE, Image.LANCZOS)

    def extract_segments(
        self,
        segment_duration: float = SEGMENT_DURATION_SEC,
        frames_per_segment: int = FRAMES_PER_SEGMENT,
    ) -> List[Dict]:
        """
        Divide the video into non-overlapping segments and sample frames.

        Returns
        -------
        List of dicts, each containing:
          - segment_id   : int (0-based)
          - start_time   : float (seconds)
          - end_time     : float (seconds)
          - frames       : list of PIL.Image (len == frames_per_segment)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError("Cannot open video for frame extraction.")

        segments: List[Dict] = []
        num_segments = int(np.ceil(self._duration / segment_duration))

        for seg_idx in range(num_segments):
            t_start = seg_idx * segment_duration
            t_end   = min((seg_idx + 1) * segment_duration, self._duration)

            f_start = int(t_start * self._fps)
            f_end   = int(t_end   * self._fps)
            f_end   = min(f_end, self._total_frames - 1)

            if f_end <= f_start:
                continue

            # Sample evenly-spaced frame indices
            indices = np.linspace(f_start, f_end - 1, frames_per_segment, dtype=int)
            frames: List[Image.Image] = []

            for idx in indices:
                img = self._read_frame(cap, idx)
                if img is not None:
                    frames.append(img)

            # Pad if we got fewer frames than expected
            if frames:
                while len(frames) < frames_per_segment:
                    frames.append(frames[-1])

                segments.append({
                    "segment_id": seg_idx,
                    "start_time": round(t_start, 2),
                    "end_time":   round(t_end, 2),
                    "frames":     frames,
                })
                logger.debug(
                    "Segment %d | %.1f–%.1f s | %d frames",
                    seg_idx, t_start, t_end, len(frames),
                )

        cap.release()
        logger.info("Extracted %d segment(s) from video.", len(segments))
        return segments


# ─── Utility functions ────────────────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    """Convert float seconds to 'MM:SS' string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """
    Quick check before processing.

    Returns (ok, message).
    """
    try:
        vp = VideoProcessor(file_path)
        vp.validate_duration()
        return True, f"Valid video — {vp.get_info()['duration_sec']} s"
    except VideoProcessingError as exc:
        return False, str(exc)
