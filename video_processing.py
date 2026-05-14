"""
video_processing.py
-------------------
Handles all video I/O operations:
- Video validation (format check, no hard duration limit)
- Frame extraction using OpenCV (memory-efficient)
- 15-second segment splitting with chunked long-video support
- YouTube video download via yt-dlp
- Thumbnail generation
"""

import cv2
import numpy as np
from PIL import Image
import logging
import os
import tempfile
from typing import List, Dict, Tuple, Optional

# ─── Logger ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_VIDEO_DURATION_SEC: float  = 600.0     # Support up to 10 minutes
SEGMENT_DURATION_SEC: float    = 15.0      # Each analysis window
FRAMES_PER_SEGMENT: int        = 16        # Frames sampled per segment
MAX_SEGMENTS_PER_CHUNK: int    = 8         # Chunk size for long videos
SUPPORTED_FORMATS: List[str]   = [".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"]
THUMBNAIL_SIZE: Tuple[int, int] = (320, 180)


# ─── Classes ─────────────────────────────────────────────────────────────────
class VideoProcessingError(Exception):
    """Raised when video cannot be read or is invalid."""


class VideoProcessor:
    """
    Handles loading, validating, and segmenting a sports video clip.

    Supports both local files and YouTube-downloaded videos.
    Uses chunked frame extraction for memory-efficient long-video processing.

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
        # Resize for memory efficiency (max 224px tall)
        h, w = rgb.shape[:2]
        if h > 224:
            scale = 224 / h
            new_w, new_h = int(w * scale), 224
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
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

    def get_duration(self) -> float:
        """Return video duration in seconds."""
        return self._duration

    def get_thumbnail(self, at_second: float = 0.5) -> Image.Image:
        """Return a single thumbnail frame resized to THUMBNAIL_SIZE."""
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = int(at_second * self._fps)
        img = self._read_frame(cap, frame_idx)
        cap.release()
        if img is None:
            img = Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 30))
        return img.resize(THUMBNAIL_SIZE, Image.LANCZOS)

    def extract_segments(
        self,
        segment_duration: float = SEGMENT_DURATION_SEC,
        frames_per_segment: int = FRAMES_PER_SEGMENT,
        max_segments: Optional[int] = None,
    ) -> List[Dict]:
        """
        Divide the video into non-overlapping segments and sample frames.
        Supports long videos through memory-efficient sequential extraction.

        Parameters
        ----------
        segment_duration   : seconds per segment
        frames_per_segment : frames sampled per segment
        max_segments       : optional cap on number of segments (for cloud mode)

        Returns
        -------
        List of dicts, each containing:
          - segment_id   : int (0-based)
          - start_time   : float (seconds)
          - end_time     : float (seconds)
          - frames       : list of PIL.Image (len == frames_per_segment)
          - duration     : float (actual segment duration)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError("Cannot open video for frame extraction.")

        segments: List[Dict] = []
        num_segments = int(np.ceil(self._duration / segment_duration))

        if max_segments:
            num_segments = min(num_segments, max_segments)

        for seg_idx in range(num_segments):
            t_start = seg_idx * segment_duration
            t_end   = min((seg_idx + 1) * segment_duration, self._duration)

            f_start = int(t_start * self._fps)
            f_end   = int(t_end   * self._fps)
            f_end   = min(f_end, self._total_frames - 1)

            if f_end <= f_start:
                continue

            # Ensure minimum duration of 4 seconds for the last segment if possible
            actual_dur = t_end - t_start
            if actual_dur < 4.0 and len(segments) > 0:
                # Append this tiny tail to the previous segment instead of creating a new one
                segments[-1]["end_time"] = round(t_end, 2)
                segments[-1]["duration"] = round(segments[-1]["end_time"] - segments[-1]["start_time"], 2)
                continue
            elif actual_dur < 1.0:
                # Too small to be a segment at all
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
                    "duration":   round(t_end - t_start, 2),
                    "frames":     frames,
                })
                logger.debug(
                    "Segment %d | %.1f–%.1f s | %d frames",
                    seg_idx, t_start, t_end, len(frames),
                )

        cap.release()
        logger.info("Extracted %d segment(s) from video.", len(segments))
        return segments

    def extract_segments_chunked(
        self,
        segment_duration: float = SEGMENT_DURATION_SEC,
        frames_per_segment: int = FRAMES_PER_SEGMENT,
        chunk_size: int = MAX_SEGMENTS_PER_CHUNK,
    ):
        """
        Generator: yields chunks of segments for memory-efficient long-video processing.
        Each chunk is a list of segment dicts (same format as extract_segments).

        Usage
        -----
        for chunk in vp.extract_segments_chunked():
            process_chunk(chunk)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError("Cannot open video for chunked extraction.")

        num_segments = int(np.ceil(self._duration / segment_duration))
        current_chunk: List[Dict] = []

        for seg_idx in range(num_segments):
            t_start = seg_idx * segment_duration
            t_end   = min((seg_idx + 1) * segment_duration, self._duration)
            f_start = int(t_start * self._fps)
            f_end   = min(int(t_end * self._fps), self._total_frames - 1)

            if f_end <= f_start:
                continue

            indices = np.linspace(f_start, f_end - 1, frames_per_segment, dtype=int)
            frames: List[Image.Image] = []

            for idx in indices:
                img = self._read_frame(cap, idx)
                if img is not None:
                    frames.append(img)

            if frames:
                while len(frames) < frames_per_segment:
                    frames.append(frames[-1])

                current_chunk.append({
                    "segment_id": seg_idx,
                    "start_time": round(t_start, 2),
                    "end_time":   round(t_end, 2),
                    "duration":   round(t_end - t_start, 2),
                    "frames":     frames,
                })

            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

        if current_chunk:
            yield current_chunk

        cap.release()


# ─── YouTube Stream Processor ────────────────────────────────────────────────
def get_youtube_stream_url(url: str) -> tuple:
    """
    Get direct stream URL from YouTube WITHOUT downloading.
    Returns (stream_url, title, duration_seconds).
    OpenCV can read frames directly from this URL.
    """
    try:
        import yt_dlp
    except ImportError:
        raise VideoProcessingError("yt-dlp not installed. Run: pip install yt-dlp")

    ydl_opts = {
        "format":      "best[ext=mp4]/best[height<=720]/best",
        "quiet":       True,
        "no_warnings": True,
        "skip_download": True,   # ← KEY: don't download, just get URL
        "extractor_args": {
            "youtube": {"player_client": ["ios", "web"]}
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.youtube.com/",
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Get the best format's direct URL
            stream_url = info.get("url")
            if not stream_url:
                # Try from formats list
                fmts = info.get("formats", [])
                # Prefer mp4, fallback to any
                for f in reversed(fmts):
                    if f.get("url") and f.get("vcodec", "none") != "none":
                        stream_url = f["url"]
                        if f.get("ext") == "mp4":
                            break
            if not stream_url:
                raise VideoProcessingError("Could not extract stream URL from YouTube.")
            title    = info.get("title", "YouTube Video")
            duration = info.get("duration", 0)
            logger.info("Stream URL obtained for: %s (%.0fs)", title, duration)
            return stream_url, title, duration
    except yt_dlp.utils.DownloadError as exc:
        raise VideoProcessingError(f"YouTube stream failed: {exc}") from exc
    except VideoProcessingError:
        raise
    except Exception as exc:
        raise VideoProcessingError(f"Unexpected stream error: {exc}") from exc


class YouTubeStreamProcessor(VideoProcessor):
    """
    VideoProcessor that reads frames directly from a YouTube stream URL.
    No file download — OpenCV reads directly from the CDN URL.
    """

    def __init__(self, stream_url: str, title: str = "YouTube Video", duration: float = 0):
        # Skip parent __init__ validation (no local file)
        self.video_path  = stream_url
        self.title       = title
        self._stream_url = stream_url
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 25.0
        self._total_frames: int = 0
        self._width:  int = 1280
        self._height: int = 720
        self._duration: float = duration
        self._load_stream_metadata()

    def _load_stream_metadata(self) -> None:
        cap = cv2.VideoCapture(self._stream_url)
        if cap.isOpened():
            self._fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self._width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fc and fc > 0:
                self._total_frames = int(fc)
                self._duration     = self._total_frames / self._fps
            cap.release()
        logger.info("Stream metadata: %.1fs | %.0f fps | %dx%d",
                    self._duration, self._fps, self._width, self._height)


# ─── Legacy download function (kept for compatibility) ────────────────────────
def download_youtube_video(url: str, output_dir: Optional[str] = None) -> str:
    """Kept for backwards compatibility. Prefer get_youtube_stream_url() instead."""
    raise VideoProcessingError(
        "Direct download disabled. Use get_youtube_stream_url() for streaming."
    )



def is_youtube_url(url: str) -> bool:
    """Return True if the URL looks like a YouTube or YouTube Shorts link."""
    import re
    patterns = [
        r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/",
        r"(https?://)?(www\.)?youtube\.com/shorts/",
    ]
    return any(re.search(p, url) for p in patterns)




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
