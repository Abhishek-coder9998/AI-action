"""
predictor.py — Football Action Predictor (Advanced)
=====================================================
Uses Microsoft X-CLIP for zero-shot video classification.

Key improvements:
- 30+ football-specific action labels (side kick, edge kick, counter attack …)
- Confidence normalization: raw softmax scores boosted to realistic 60–92% range
- Temperature scaling on logits for sharper predictions
- Top-K scoring with relative ranking preserved
"""

import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─── Extended Football Action Label Bank ─────────────────────────────────────
# Format: short_label → descriptive prompt (CLIP-family models prefer sentences)

FOOTBALL_ACTIONS: Dict[str, str] = {
    # ── Attacking ────────────────────────────────────────────────────────────
    "Goal Scored":            "a football player scores a goal and the ball enters the net",
    "Shot on Target":         "a football player shoots the ball powerfully towards the goal",
    "Goal Attempt":           "a football player attempts to kick the ball towards the goal",
    "Side Kick":              "a football player kicks the ball sideways with the side of their foot",
    "Edge Kick":              "a football player kicks the ball sharply from a tight edge angle",
    "Penalty Kick":           "a football player takes a penalty kick from the penalty spot",
    "Free Kick":              "a football player takes a direct free kick aiming at goal",
    "Attacking Transition":   "football players rapidly move the ball forward towards the opponent's goal",
    "Counter Attack":         "football players launch a fast counter attack breaking forward quickly",
    "Through Pass":           "a football player plays a precise through pass splitting the defence",
    "Cross":                  "a football player crosses the ball from the wing into the penalty area",
    "Crossing Movement":      "a football player makes a wide run and delivers a crossing ball",
    "Header":                 "a football player jumps and heads the ball with their forehead",
    "Dribble":                "a football player dribbles past defenders with skill and speed",

    # ── Defensive ────────────────────────────────────────────────────────────
    "Tackle / Sliding":       "a football player makes a sliding tackle to win the ball",
    "Sliding Interception":   "a football player slides across the ground to intercept the ball",
    "Defensive Clearance":    "a football player clears the ball away from danger near their own goal",
    "Defending Ball":         "a football player positions their body to block and protect the ball",
    "Defensive Interception": "a football defender intercepts a pass and wins possession",
    "Ball Recovery":          "a football player recovers the ball after a loose play situation",
    "Aerial Duel":            "two football players jump and compete in the air to win a header",
    "Aggressive Pressing":    "multiple football defenders close down the ball aggressively as a unit",

    # ── Set Pieces ────────────────────────────────────────────────────────────
    "Corner Kick":            "a football player takes a corner kick from the corner flag area",
    "Throw-in":               "a football player throws the ball in from the sideline",

    # ── Goalkeeper ───────────────────────────────────────────────────────────
    "Goalkeeper Save":        "a goalkeeper dives and saves the ball preventing a goal",

    # ── Fouls / Discipline ───────────────────────────────────────────────────
    "Foul":                   "a football player commits a foul by illegally challenging an opponent",
    "Offside":                "a football player is caught in an offside position",
    "Yellow / Red Card":      "a referee shows a yellow or red card to a football player",

    # ── General Match ────────────────────────────────────────────────────────
    "Pass":                   "a football player passes the ball accurately to a teammate",
    "Celebration":            "football players celebrate together after scoring a goal",
    "Ball in Play":           "football players are actively playing and competing for the ball",
}

# Optional general sports overlay
GENERAL_SPORT_EXTRAS: Dict[str, str] = {
    "Sprint":      "an athlete sprints at maximum speed across the field",
    "Jump":        "an athlete jumps powerfully into the air",
    "Block":       "an athlete blocks an opponent's action with their body",
    "Score Point": "an athlete scores a point for their team",
}

# Action categories for UI grouping
ACTION_CATEGORIES: Dict[str, List[str]] = {
    "⚡ Attacking":    ["Goal Scored", "Shot on Target", "Goal Attempt", "Side Kick",
                        "Edge Kick", "Penalty Kick", "Free Kick", "Attacking Transition",
                        "Counter Attack", "Through Pass", "Cross", "Crossing Movement",
                        "Header", "Dribble"],
    "🛡️ Defending":   ["Tackle / Sliding", "Sliding Interception", "Defensive Clearance",
                        "Defending Ball", "Defensive Interception", "Ball Recovery",
                        "Aerial Duel", "Aggressive Pressing"],
    "🎯 Set Pieces":  ["Corner Kick", "Throw-in", "Penalty Kick", "Free Kick"],
    "🧤 Goalkeeper":  ["Goalkeeper Save"],
    "🚫 Discipline":  ["Foul", "Offside", "Yellow / Red Card"],
    "⚽ General":     ["Pass", "Celebration", "Ball in Play"],
}


# ─── Confidence Normalizer ────────────────────────────────────────────────────
class ConfidenceNormalizer:
    """
    Maps raw X-CLIP softmax probabilities (typically 3–20% with 30 labels)
    to realistic football intelligence confidence scores (60–93%).

    Method:
    1. Apply temperature scaling on raw logits → sharper distribution
    2. Rescale top-K probabilities to [MIN_DISPLAY, MAX_DISPLAY] range
    3. Preserve relative ranking between actions
    """

    MIN_DISPLAY: float = 0.25   # lowered from 0.38 to detect more actions
    MAX_DISPLAY: float = 0.93   
    TEMPERATURE: float = 0.12   # slightly higher to allow more variation

    @classmethod
    def normalize(cls, raw_probs: np.ndarray, top_k: int = 6) -> np.ndarray:
        """Rescale raw probability vector to display-friendly range."""
        if len(raw_probs) == 0:
            return raw_probs

        # Step 1: Convert probabilities back to log-space and apply temperature
        log_probs = np.log(np.clip(raw_probs, 1e-9, 1.0))
        scaled    = log_probs / cls.TEMPERATURE
        # Re-normalize via softmax
        exp_s     = np.exp(scaled - np.max(scaled))
        sharpened = exp_s / exp_s.sum()

        # Step 2: Min-max rescale to [MIN_DISPLAY, MAX_DISPLAY]
        top_indices = np.argsort(sharpened)[::-1][:top_k]
        top_vals    = sharpened[top_indices]
        v_min, v_max = top_vals[-1], top_vals[0]

        normalized = np.zeros_like(sharpened)
        for idx in top_indices:
            if v_max == v_min:
                normalized[idx] = cls.MAX_DISPLAY
            else:
                t = (sharpened[idx] - v_min) / (v_max - v_min)
                normalized[idx] = cls.MIN_DISPLAY + t * (cls.MAX_DISPLAY - cls.MIN_DISPLAY)

        return normalized


# ─── Main Predictor ───────────────────────────────────────────────────────────
class FootballActionPredictor:
    """
    Zero-shot football action recognition using Microsoft X-CLIP.

    Parameters
    ----------
    device    : 'cuda' | 'cpu' | None (auto)
    sport_mode: 'football' | 'general'
    top_k     : number of top actions returned per segment
    """

    MODEL_NAME = "microsoft/xclip-base-patch32"

    def __init__(
        self,
        device: Optional[str] = None,
        sport_mode: str = "football",
        top_k: int = 6,
    ):
        import torch

        self.torch      = torch
        self.device     = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.sport_mode = sport_mode
        self.top_k      = top_k
        self._normalizer = ConfidenceNormalizer()
        self._load_model()
        self._build_label_bank()

    def _load_model(self) -> None:
        from transformers import XCLIPProcessor, XCLIPModel

        logger.info("Loading %s on %s …", self.MODEL_NAME, self.device)
        self.processor = XCLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model     = XCLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        logger.info("Model ready ✓")

    def _build_label_bank(self) -> None:
        base = dict(FOOTBALL_ACTIONS)
        if self.sport_mode == "general":
            base.update(GENERAL_SPORT_EXTRAS)
        self._label_bank    = base
        self._short_labels  = list(base.keys())
        self._prompts       = list(base.values())
        logger.info("Label bank: %d actions.", len(self._short_labels))

    def _sample_frames(self, frames: List[Image.Image], n: int = 8) -> List[Image.Image]:
        if len(frames) == n:
            return frames
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        sampled = [frames[i] for i in indices]
        while len(sampled) < n:
            sampled.append(sampled[-1])
        return sampled

    @property
    def available_labels(self) -> List[str]:
        return self._short_labels

    # ── ImageNet normalization constants (used by X-CLIP) ────────────────────
    _MEAN = [0.48145466, 0.4578275,  0.40821073]
    _STD  = [0.26862954, 0.26130258, 0.27577711]

    def _frames_to_tensor(self, frames: List[Image.Image]) -> "torch.Tensor":
        """
        Manually convert PIL frames → float32 tensor of shape (1, T, 3, 224, 224).
        Bypasses XCLIPProcessor video path to avoid shape mismatches.
        """
        import numpy as np
        mean = np.array(self._MEAN, dtype=np.float32)
        std  = np.array(self._STD,  dtype=np.float32)
        processed = []
        for img in frames:
            img_rgb = img.convert("RGB").resize((224, 224), Image.LANCZOS)
            arr = np.array(img_rgb, dtype=np.float32) / 255.0
            arr = (arr - mean) / std          # HWC
            arr = arr.transpose(2, 0, 1)      # CHW
            processed.append(arr)
        # Stack → (T, C, H, W) then unsqueeze → (1, T, C, H, W)
        video_np = np.stack(processed, axis=0)
        return self.torch.tensor(video_np).unsqueeze(0)

    def predict_segment(self, frames: List[Image.Image]) -> List[Dict]:
        """
        Classify actions in one video segment.
        Returns list of top-K dicts: { label, confidence, rank }
        """
        sampled = self._sample_frames(frames, n=8)

        # ── Text encoding via processor ───────────────────────────────────────
        text_inputs = self.processor(
            text=self._prompts,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # ── Video encoding — manually built tensor ────────────────────────────
        pixel_values = self._frames_to_tensor(sampled).to(self.device)
        # X-CLIP forward: pixel_values shape must be (B, T, C, H, W)
        inputs = {**text_inputs, "pixel_values": pixel_values}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        logits    = outputs.logits_per_video         # [1, num_labels]
        raw_probs = logits.softmax(dim=1).cpu().numpy()[0]

        # Normalize confidence to realistic range
        norm_probs = self._normalizer.normalize(raw_probs, top_k=self.top_k)

        # Build results dict
        all_scores = {
            lbl: float(norm_probs[i]) if norm_probs[i] > 0 else float(raw_probs[i])
            for i, lbl in enumerate(self._short_labels)
        }

        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"label": lbl, "confidence": conf, "rank": i + 1}
            for i, (lbl, conf) in enumerate(sorted_scores[: self.top_k])
        ]


    def predict_all_segments(
        self,
        segments: List[Dict],
        progress_callback=None,
    ) -> List[Dict]:
        """Run prediction on all extracted video segments."""
        results: List[Dict] = []
        total = len(segments)

        for i, seg in enumerate(segments):
            top_k_preds = self.predict_segment(seg["frames"])
            best = top_k_preds[0] if top_k_preds else {"label": "Ball in Play", "confidence": 0.25}

            results.append({
                "segment_id":     seg["segment_id"],
                "start_time":     seg["start_time"],
                "end_time":       seg["end_time"],
                "top_action":     best["label"],
                "top_confidence": best["confidence"],
                "top_k_actions":  top_k_preds,
            })

            if progress_callback:
                progress_callback(i + 1, total)

            logger.info(
                "Seg %d [%.0f–%.0f s] → %s (%.0f%%)",
                seg["segment_id"], seg["start_time"], seg["end_time"],
                best["label"], best["confidence"] * 100,
            )

        return results


# Backwards-compatible alias
class ActionPredictor(FootballActionPredictor):
    def predict_segments_batch(self, segments, labels=None, callback=None):
        return self.predict_all_segments(segments, progress_callback=callback)
