import cv2
import numpy as np
from PIL import Image
import logging
import os
import tempfile
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MAX_VIDEO_DURATION_SEC: float  = 600.0
SEGMENT_DURATION_SEC: float    = 4.0  # Slightly smaller for more diversity
SUPPORTED_FORMATS: List[str]   = [".mp4", ".mov", ".avi", ".webm", ".mkv"]
THUMBNAIL_SIZE: Tuple[int, int] = (320, 180)

class VideoProcessingError(Exception):
    pass

class VideoProcessor:
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise VideoProcessingError(f"File not found: {video_path}")
        self.video_path = video_path
        self._load_metadata()

    def _load_metadata(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"OpenCV could not open video: {self.video_path}")
        self._fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration     = self._total_frames / self._fps
        cap.release()
        logger.info(f"Video loaded: {self._duration:.1f}s | {self._fps:.0f}fps")

    def get_info(self) -> Dict:
        return {
            "path": self.video_path,
            "duration_sec": round(self._duration, 2),
            "fps": round(self._fps, 2),
            "total_frames": self._total_frames,
        }

    def get_duration(self) -> float:
        return self._duration

    def extract_segments(
        self,
        segment_duration: float = SEGMENT_DURATION_SEC,
        sampling_rate: float = 0.5, # 1 frame every 0.5 seconds
    ) -> List[Dict]:
        """
        Extracts frames and ensures min 2.0s gap between segments.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoProcessingError("Cannot open video for frame extraction.")

        segments: List[Dict] = []
        # Force at least 7 segments if video is long enough
        if self._duration > 14.0:
            segment_duration = min(segment_duration, self._duration / 7.0)
            
        num_segments = int(np.ceil(self._duration / segment_duration))
        
        total_extracted_frames = 0
        prev_frame_gray = None

        for seg_idx in range(num_segments):
            t_start = seg_idx * segment_duration
            t_end   = min((seg_idx + 1) * segment_duration, self._duration)

            # Ensure minimum 2.0s gap
            if (t_end - t_start) < 2.0 and seg_idx > 0:
                segments[-1]["end_time"] = round(t_end, 2)
                segments[-1]["duration"] = round(segments[-1]["end_time"] - segments[-1]["start_time"], 2)
                continue

            timestamps = np.arange(t_start, t_end, sampling_rate)
            frames_data = []
            seg_motion_mags = []
            seg_motion_dirs = []

            for ts in timestamps:
                f_idx = int(ts * self._fps)
                if f_idx >= self._total_frames: break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret: continue
                
                # Optical Flow Calculation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    avg_mag = np.mean(mag) * 10
                    seg_motion_mags.append(avg_mag)
                    
                    avg_ang = np.mean(ang) * 180 / np.pi
                    if 0 <= avg_ang < 45 or 315 <= avg_ang <= 360: dir_name = "forward attack"
                    elif 135 <= avg_ang < 225: dir_name = "backward defense"
                    else: dir_name = "lateral"
                    seg_motion_dirs.append(dir_name)
                
                prev_frame_gray = gray
                
                # Resize for model input efficiency
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (320, 240))
                pil_img = Image.fromarray(rgb_resized)
                frames_data.append(pil_img)
                total_extracted_frames += 1

            if frames_data:
                # Minimum 8 frames for X-CLIP
                while len(frames_data) < 8: frames_data.append(frames_data[-1])
                
                avg_mag = np.mean(seg_motion_mags) if seg_motion_mags else 0
                dom_dir = max(set(seg_motion_dirs), key=seg_motion_dirs.count) if seg_motion_dirs else "unknown"

                segments.append({
                    "segment_id": len(segments),
                    "start_time": round(t_start, 2),
                    "end_time":   round(t_end, 2),
                    "duration":   round(t_end - t_start, 2),
                    "frames":     frames_data,
                    "motion_data": {
                        "magnitude": avg_mag,
                        "direction": dom_dir
                    }
                })

        cap.release()
        print(f"[DEBUG] Extracted {total_extracted_frames} frames from {self._duration:.1f} second video")
        return segments

    def get_thumbnail(self, at_second: float = 0.5) -> Image.Image:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_second * self._fps))
        ret, frame = cap.read()
        cap.release()
        if not ret: return Image.new("RGB", THUMBNAIL_SIZE, (30, 30, 30))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).resize(THUMBNAIL_SIZE, Image.LANCZOS)

def is_youtube_url(url: str) -> bool:
    import re
    return any(re.search(p, url) for p in [r"youtube\.com/", r"youtu\.be/"])

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

# Mock for YouTube streaming (already implemented in project, keeping basic structure)
def get_youtube_stream_url(url: str):
    return url, "YouTube Video", 0
