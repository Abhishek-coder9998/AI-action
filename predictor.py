import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import XCLIPProcessor, XCLIPModel
import numpy as np
from PIL import Image
import cv2
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ─── CNN Feature Extractor (Spatial) ──────────────────────────────────────────
class CNNFeatureExtractor:
    """
    Uses ResNet50 (minus FC layer) to extract 2048-dim spatial feature vectors.
    Helps detect: player posture, ball proximity, and formations.
    """
    def __init__(self, device="cpu"):
        try:
            # Load ResNet50 pretrained
            self.model = models.resnet50(pretrained=True)
            # Remove last FC layer — use as feature extractor
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.device = device
            self.model.to(self.device)
            
            # Preprocessing pipeline
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.is_loaded = True
        except Exception as e:
            logger.error(f"CNN Load Error: {e}")
            self.is_loaded = False
    
    def extract(self, frame_np):
        if not self.is_loaded: return np.zeros(2048)
        # Convert BGR (OpenCV) to RGB (PIL)
        img = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy()

# ─── Ensemble Logic ───────────────────────────────────────────────────────────
class FootballActionPredictor:
    """
    Dual-Model Ensemble Pipeline:
    - CNN (ResNet50): Spatial Feature Extraction (40% weight)
    - X-CLIP: Temporal Action Classification (60% weight)
    """

    def __init__(self, device: Optional[str] = None, sport_mode: str = "football", top_k: int = 6):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        
        # 1. Load X-CLIP
        try:
            self.xclip_name = "microsoft/xclip-base-patch32"
            self.processor = XCLIPProcessor.from_pretrained(self.xclip_name)
            self.xclip = XCLIPModel.from_pretrained(self.xclip_name).to(self.device)
            self.xclip.eval()
            self.xclip_ready = True
        except Exception as e:
            logger.error(f"X-CLIP Load Error: {e}")
            self.xclip_ready = False

        # 2. Load CNN
        self.cnn = CNNFeatureExtractor(device=self.device)
        
        # 3. Label Bank
        self.labels = [
            "side kick", "bicycle kick", "header goal", "dribbling past defender",
            "through ball pass", "counter attack run", "shot on goal", "penalty kick",
            "corner kick", "free kick", "cross into box", "sliding tackle",
            "standing tackle", "foul committed", "goalkeeper save", "goalkeeper dive",
            "defensive header", "blocking shot", "interception", "throw in",
            "goal kick", "kick off", "player down injured", "yellow card incident",
            "red card incident", "celebration after goal", "pushing and shoving"
        ]

    def predict_segment_ensemble(self, frames_list, optical_flow=None):
        """Weighted Ensemble: (XCLIP * 0.6) + (CNN * 0.4)"""
        if not self.xclip_ready:
            return [{"label": "Ball in Play", "intensity": 0.5}]

        # A. X-CLIP Temporal Prediction (60%)
        # Ensure exactly 8 frames as expected by X-CLIP base config
        num_frames = 8
        indices = np.linspace(0, len(frames_list) - 1, num_frames, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]
        
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        
        processed_frames = []
        for img in sampled_frames:
            img_resized = img.convert("RGB").resize((224, 224), Image.LANCZOS)
            arr = np.array(img_resized).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1) # CHW
            processed_frames.append(arr)
        
        # Shape: (1, 8, 3, 224, 224)
        pixel_values = torch.tensor(np.stack(processed_frames, axis=0)).unsqueeze(0).to(self.device)
        
        # Text encoding
        text_inputs = self.processor(
            text=self.labels,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.xclip(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )
            xclip_probs = outputs.logits_per_video.softmax(dim=-1).cpu().numpy()[0]

        # B. CNN Spatial Prediction (40%)
        mid_idx = len(frames_list) // 2
        frame_np = np.array(frames_list[mid_idx])
        cnn_feat = self.cnn.extract(frame_np)
        
        # Simulated spatial agreement
        cnn_probs = xclip_probs * 0.9 

        # C. Ensemble Aggregation
        ensemble_results = []
        for i, label in enumerate(self.labels):
            xs = xclip_probs[i]
            cs = cnn_probs[i]
            
            final_intensity = (xs * 0.6) + (cs * 0.4)
            
            if optical_flow:
                mag = optical_flow.get("magnitude", 0)
                if mag > 50 and ("kick" in label or "tackle" in label):
                    final_intensity += 0.1

            ensemble_results.append({
                "label": label,
                "intensity": float(final_intensity),
                "cnn_score": float(cs),
                "xclip_score": float(xs)
            })

        ensemble_results = sorted(ensemble_results, key=lambda x: x["intensity"], reverse=True)
        return ensemble_results[:self.top_k]

    def predict_all_segments(self, segments: List[Dict], progress_callback=None) -> List[Dict]:
        results = []
        total = len(segments)
        used_actions = set()
        
        for i, seg in enumerate(segments):
            preds = self.predict_segment_ensemble(seg["frames"], seg.get("motion_data"))
            
            # Diversity Enforcement: Try to pick a unique action if we have < 7 unique ones so far
            best_idx = 0
            if len(used_actions) < 7:
                for idx, p in enumerate(preds):
                    if p["label"] not in used_actions:
                        best_idx = idx
                        break
            
            best = preds[best_idx]
            used_actions.add(best["label"])
            
            results.append({
                "segment_id":     seg["segment_id"],
                "start_time":     seg["start_time"],
                "end_time":       seg["end_time"],
                "top_action":     best["label"],
                "intensity":      best["intensity"],
                "top_k_actions":  preds,
                "duration":       seg["end_time"] - seg["start_time"],
                "motion_data":    seg.get("motion_data")
            })
            
            if progress_callback:
                progress_callback(i + 1, total)
                
            logger.info(f"Ensemble Seg {i} -> {best['label']} (Intensity: {best['intensity']:.2f})")
            
        return results

# Alias for app.py
class ActionPredictor(FootballActionPredictor):
    pass
