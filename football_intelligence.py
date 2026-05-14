"""
football_intelligence.py
========================
Optical-flow & motion-based football intelligence layer.

Uses OpenCV to analyze raw frames and overlay football-specific heuristics:
  - Motion magnitude → high vs low activity
  - Motion direction → ball trajectory (side / edge / forward)
  - Region of interest → goalkeeper area, midfield, attack zone
  - Density of motion → pressing, counter-attack, defensive shape

No external ML model needed — pure OpenCV.  Works offline.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ─── Zone definitions (fractional coords of frame width/height) ──────────────
ZONE_DEFS: Dict[str, Tuple[float, float, float, float]] = {
    # x_min, y_min, x_max, y_max  (all 0–1)
    "left_wing":   (0.00, 0.0, 0.25, 1.0),
    "right_wing":  (0.75, 0.0, 1.00, 1.0),
    "midfield":    (0.25, 0.2, 0.75, 0.8),
    "penalty_box": (0.20, 0.1, 0.80, 0.5),
    "goal_area":   (0.30, 0.0, 0.70, 0.2),
}


def pil_to_gray(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV grayscale uint8 array."""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def compute_optical_flow(
    frame1: Image.Image,
    frame2: Image.Image,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dense Farneback optical flow between two consecutive frames.

    Returns
    -------
    flow     : (H, W, 2) float32 — (dx, dy) per pixel
    magnitude: (H, W) float32 — pixel motion strength
    angle    : (H, W) float32 — pixel motion direction in degrees
    """
    g1 = pil_to_gray(frame1)
    g2 = pil_to_gray(frame2)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return flow, magnitude, angle


def analyze_motion_signature(
    frames: List[Image.Image],
) -> Dict:
    """
    Analyze motion patterns across a segment's frames.

    Parameters
    ----------
    frames : list of PIL.Image (any length ≥ 2)

    Returns
    -------
    dict with keys:
      avg_magnitude     : float  (overall motion level)
      dominant_direction: str    (cardinal direction of main motion)
      lateral_ratio     : float  (left-right vs up-down motion fraction)
      zone_activity     : dict   (activity level per pitch zone)
      motion_class      : str    (heuristic label: 'high', 'medium', 'low')
      football_hints    : list   (list of inferred football intelligence strings)
    """
    if len(frames) < 2:
        return _empty_signature()

    magnitudes: List[float] = []
    dx_list:    List[float] = []
    dy_list:    List[float] = []
    zone_sums:  Dict[str, float] = {z: 0.0 for z in ZONE_DEFS}

    for i in range(min(len(frames) - 1, 6)):   # analyse up to 6 pairs
        try:
            _, mag, ang = compute_optical_flow(frames[i], frames[i + 1])
        except Exception:
            continue

        h, w = mag.shape
        magnitudes.append(float(mag.mean()))

        # Decompose flow
        flow_x = mag * np.cos(np.deg2rad(ang))
        flow_y = mag * np.sin(np.deg2rad(ang))
        dx_list.append(float(flow_x.mean()))
        dy_list.append(float(flow_y.mean()))

        # Zone activity
        for zone, (x0, y0, x1, y1) in ZONE_DEFS.items():
            r0, r1 = int(y0 * h), int(y1 * h)
            c0, c1 = int(x0 * w), int(x1 * w)
            zone_sums[zone] += float(mag[r0:r1, c0:c1].mean())

    if not magnitudes:
        return _empty_signature()

    avg_mag  = float(np.mean(magnitudes))
    avg_dx   = float(np.mean(dx_list))
    avg_dy   = float(np.mean(dy_list))

    # Normalize zone activity
    n_pairs = len(magnitudes)
    zone_activity = {z: v / n_pairs for z, v in zone_sums.items()}

    # Dominant motion direction
    dominant_dir = _direction_label(avg_dx, avg_dy)

    # Lateral ratio (how sideways vs forward motion is)
    total_motion = abs(avg_dx) + abs(avg_dy) + 1e-6
    lateral_ratio = abs(avg_dx) / total_motion

    # Motion class
    if avg_mag > 6.0:
        motion_class = "high"
    elif avg_mag > 2.5:
        motion_class = "medium"
    else:
        motion_class = "low"

    # Generate football heuristic hints
    hints = _generate_football_hints(
        avg_mag, lateral_ratio, dominant_dir, zone_activity, motion_class
    )

    return {
        "avg_magnitude":      avg_mag,
        "dominant_direction": dominant_dir,
        "lateral_ratio":      lateral_ratio,
        "zone_activity":      zone_activity,
        "motion_class":       motion_class,
        "football_hints":     hints,
    }


def _direction_label(dx: float, dy: float) -> str:
    angle = np.degrees(np.arctan2(dy, dx))
    if   -45 <= angle < 45:   return "right"
    elif  45 <= angle < 135:  return "down"
    elif angle >= 135 or angle < -135: return "left"
    else:                      return "up"


def _generate_football_hints(
    magnitude: float,
    lateral_ratio: float,
    direction: str,
    zones: Dict[str, float],
    motion_class: str,
) -> List[str]:
    hints: List[str] = []

    # ── High lateral motion → side / edge kick
    if lateral_ratio > 0.65 and magnitude > 3.0:
        if direction in ("left", "right"):
            hints.append("Side Kick")
        else:
            hints.append("Cross")

    # ── Acute angle motion → edge kick
    if 0.50 < lateral_ratio < 0.65 and direction in ("left", "right"):
        hints.append("Edge Kick")

    # ── High motion overall
    if motion_class == "high":
        if zones.get("penalty_box", 0) > zones.get("midfield", 0):
            hints.append("Goal Attempt")
            hints.append("Attacking Transition")
        elif lateral_ratio < 0.4: # mostly forward/backward
            hints.append("Counter Attack")
            hints.append("Sprint")
        else:
            hints.append("Dribble")

    # ── Midfield heavy → passing / possession
    if zones.get("midfield", 0) > 4.0 and motion_class in ("medium", "high"):
        if lateral_ratio > 0.5:
            hints.append("Through Pass")
        else:
            hints.append("Pass Sequence")

    # ── Wing-heavy motion → crossing
    wing_activity = zones.get("left_wing", 0) + zones.get("right_wing", 0)
    mid_activity  = zones.get("midfield", 0)
    if wing_activity > mid_activity * 1.1:
        hints.append("Crossing Movement")
        hints.append("Cross")

    # ── Goal area activity → goalkeeper save or goal
    if zones.get("goal_area", 0) > 3.0:
        if magnitude > 5.0:
            hints.append("Goal Scored")
        else:
            hints.append("Goalkeeper Save")

    # ── Low / static motion → defending / set pieces
    if motion_class == "low":
        if zones.get("penalty_box", 0) > 2.0:
            hints.append("Offside Trap")
        else:
            hints.append("Defending Ball")
            hints.append("Tackle / Sliding")

    # ── Dense mid-field motion → pressing
    if zones.get("midfield", 0) > 6.0:
        hints.append("Aggressive Pressing")

    return list(dict.fromkeys(hints))   # de-duplicate preserving order


def _empty_signature() -> Dict:
    return {
        "avg_magnitude": 0.0,
        "dominant_direction": "unknown",
        "lateral_ratio": 0.0,
        "zone_activity": {},
        "motion_class": "low",
        "football_hints": [],
    }


def build_heatmap_data(frames: List[Image.Image]) -> Optional[np.ndarray]:
    """
    Produce a motion heatmap (H × W float array) from a list of frames.
    Used for the Plotly heatmap visualization in the UI.
    """
    if len(frames) < 2:
        return None

    heatmap: Optional[np.ndarray] = None

    for i in range(min(len(frames) - 1, 8)):
        try:
            _, mag, _ = compute_optical_flow(frames[i], frames[i + 1])
            if heatmap is None:
                heatmap = mag.astype(np.float32)
            else:
                heatmap += mag.astype(np.float32)
        except Exception:
            continue

    if heatmap is not None:
        # Normalize 0–1
        hmax = heatmap.max()
        if hmax > 0:
            heatmap /= hmax
        # Downsample for UI performance
        heatmap = cv2.resize(heatmap, (160, 90))

    return heatmap


def boost_confidence_with_hints(
    top_k_results: List[Dict],
    football_hints: List[str],
    boost: float = 0.08,
) -> List[Dict]:
    """
    Slightly boost confidence of predictions that match motion-based hints.
    Clamps at 0.97 to keep realistic.
    """
    hint_set = set(football_hints)
    for item in top_k_results:
        if item["label"] in hint_set:
            item["confidence"] = min(item["confidence"] + boost, 0.97)
    return top_k_results
