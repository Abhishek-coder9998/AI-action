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
SUPPORTED_FORMATS: List[str]   = [".mp4", ".mov", ".avi", ".webm", ".mkv"]
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


# ─── YouTube Downloader ───────────────────────────────────────────────────────
def download_youtube_video(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download a YouTube video (standard, Shorts, or MP4 link) using yt-dlp.

    Parameters
    ----------
    url        : YouTube URL (standard, Shorts, or MP4)
    output_dir : directory to save the file (defaults to system temp dir)

    Returns
    -------
    str : absolute path to the downloaded MP4 file.

    Raises
    ------
    VideoProcessingError : if download fails or yt-dlp is not installed.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise VideoProcessingError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    out_template = os.path.join(output_dir, "yt_%(id)s.%(ext)s")

    ydl_opts = {
        "format":       "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":      out_template,
        "quiet":        True,
        "no_warnings":  True,
        "merge_output_format": "mp4",
    }

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "video")
            # Find the downloaded file
            downloaded = os.path.join(output_dir, f"yt_{video_id}.mp4")
            if not os.path.exists(downloaded):
                # Fallback: search for any matching file
                for f in os.listdir(output_dir):
                    if f.startswith(f"yt_{video_id}") and f.endswith(".mp4"):
                        downloaded = os.path.join(output_dir, f)
                        break
            if not os.path.exists(downloaded):
                raise VideoProcessingError(
                    f"Download appeared to succeed but file not found: {downloaded}"
                )
            logger.info("YouTube video downloaded: %s", downloaded)
            return downloaded
    except yt_dlp.utils.DownloadError as exc:
        raise VideoProcessingError(f"YouTube download failed: {exc}") from exc
    except Exception as exc:
        raise VideoProcessingError(f"Unexpected error during download: {exc}") from exc


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
