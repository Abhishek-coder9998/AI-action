import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import random

# ─── Action Color & Icon Mapping ─────────────────────────────────────────────
ACTION_META = {
    "side kick":              {"color": "#EF4444", "icon": "⚽"},
    "bicycle kick":           {"color": "#EF4444", "icon": "⚽"},
    "header goal":            {"color": "#F97316", "icon": "👤"},
    "dribbling past defender": {"color": "#EAB308", "icon": "💫"},
    "through ball pass":      {"color": "#14B8A6", "icon": "➡️"},
    "cross into box":         {"color": "#14B8A6", "icon": "➡️"},
    "counter attack run":     {"color": "#10B981", "icon": "⚡"},
    "shot on goal":           {"color": "#EF4444", "icon": "⚽"},
    "penalty kick":           {"color": "#3B82F6", "icon": "🎯"},
    "corner kick":            {"color": "#3B82F6", "icon": "🎯"},
    "free kick":              {"color": "#3B82F6", "icon": "🎯"},
    "sliding tackle":         {"color": "#06B6D4", "icon": "🛡️"},
    "standing tackle":        {"color": "#06B6D4", "icon": "🛡️"},
    "foul committed":         {"color": "#F43F5E", "icon": "⚠️"},
    "goalkeeper save":        {"color": "#8B5CF6", "icon": "🤚"},
    "goalkeeper dive":        {"color": "#8B5CF6", "icon": "🤚"},
    "defensive header":       {"color": "#06B6D4", "icon": "🛡️"},
    "blocking shot":          {"color": "#06B6D4", "icon": "🛡️"},
    "interception":           {"color": "#06B6D4", "icon": "✋"},
    "throw in":               {"color": "#64748B", "icon": "👐"},
    "goal kick":              {"color": "#8B5CF6", "icon": "🤚"},
    "kick off":               {"color": "#10B981", "icon": "⚽"},
    "player down injured":    {"color": "#F43F5E", "icon": "⚠️"},
    "yellow card incident":   {"color": "#FBBF24", "icon": "🟨"},
    "red card incident":      {"color": "#EF4444", "icon": "🟥"},
    "celebration after goal": {"color": "#EC4899", "icon": "🎉"},
    "pushing and shoving":    {"color": "#F43F5E", "icon": "⚠️"},
}

ACTION_COMMENTARY = {
    "side kick": "Brilliant technique! Powerful side kick unleashed!",
    "bicycle kick": "Spectacular acrobatic bicycle kick attempt!",
    "header goal": "Towering header rises above everyone to meet the cross!",
    "dribbling past defender": "Silky footwork glides past the defender!",
    "through ball pass": "Inch-perfect through ball splits the defense!",
    "counter attack run": "Lightning fast break — surging forward on the counter!",
    "shot on goal": "Fierce strike at goal — keeper must be at his best!",
    "penalty kick": "Referee points to the spot — penalty awarded!",
    "corner kick": "Dangerous ball whipped in from the corner!",
    "free kick": "Dead ball — wall is set, ready to curl it in!",
    "cross into box": "Dangerous cross swings into the penalty area!",
    "sliding tackle": "Crunching sliding tackle perfectly timed!",
    "standing tackle": "Strong standing tackle cuts out the attack!",
    "foul committed": "Referee stops play — clear foul committed!",
    "goalkeeper save": "What a save! Goalkeeper denies brilliantly!",
    "goalkeeper dive": "Full stretch dive — fingertips save!",
    "defensive header": "Commanding header clears the danger!",
    "blocking shot": "Last ditch block — throws body in front!",
    "interception": "Sharp interception reads the pass perfectly!",
    "throw in": "Play restarts with throw in from touchline.",
    "goal kick": "Goalkeeper distributes from the back.",
    "kick off": "Whistle blows — kick off starts the action!",
    "player down injured": "Play stops — player down receiving treatment.",
    "yellow card incident": "Referee reaches for the yellow card!",
    "red card incident": "Straight red — player is sent off!",
    "celebration after goal": "GOAL! Players celebrate — what a moment!",
    "pushing and shoving": "Tempers flare — pushing and shoving on the pitch!",
}

def get_action_meta(action: str) -> Dict:
    return ACTION_META.get(action.lower(), {"color": "#64748B", "icon": "⚽"})

def generate_smart_commentary(action: str) -> str:
    return ACTION_COMMENTARY.get(action.lower(), "Tactical execution detected in this phase of the match.")

def generate_tags(action: str) -> List[str]:
    # Dynamic Hashtag Generation based on action type
    tags = ["#Football", "#Analytics"]
    action = action.lower()
    if "kick" in action or "shot" in action: tags = ["#Attack", "#Kick", "#Shot"]
    elif "tackle" in action or "block" in action: tags = ["#Defense", "#Challenge", "#Solid"]
    elif "save" in action or "dive" in action: tags = ["#Goalie", "#Save", "#Reflex"]
    elif "pass" in action or "cross" in action: tags = ["#Playmaking", "#Vision", "#KeyPass"]
    elif "foul" in action or "card" in action: tags = ["#Referee", "#Discipline", "#Incident"]
    return tags

def build_heatmap_data(frames: List[Image.Image]) -> np.ndarray:
    # Generates a dummy heatmap structure for UI rendering
    # In production, this uses real Optical Flow data from video_processing.py
    heatmap = np.zeros((90, 160))
    # Add some random activity centers
    for _ in range(5):
        cy, cx = random.randint(10, 80), random.randint(10, 150)
        heatmap[cy-5:cy+5, cx-5:cx+5] += random.random()
    return heatmap
