"""
commentary.py — Professional Football Commentary & Match Analysis
=================================================================
Generates:
  1. Broadcast-style per-event commentary
  2. Professional AI match summary paragraph
  3. Match Analysis Card (tactical breakdown)
"""

import random
from typing import List, Dict
from collections import Counter

# ─── Commentary Templates ────────────────────────────────────────────────────
COMMENTARY_TEMPLATES: Dict[str, List[str]] = {
    "Goal Scored": [
        "GOAL! An unstoppable finish — the ball rockets into the back of the net!",
        "GOOOAL! Clinical finishing from close range — the goalkeeper had no chance!",
        "It's in! The striker capitalises on a defensive lapse to score!",
        "Phenomenal! A perfectly placed shot finds the bottom corner!",
    ],
    "Shot on Target": [
        "A ferocious strike forces the goalkeeper into a superb diving save!",
        "The attacker unleashes a thunderous shot — right on target!",
        "Excellent attempt on goal — the keeper is fully stretched!",
        "A powerful, low drive tests the goalkeeper's reactions!",
    ],
    "Goal Attempt": [
        "The striker lines up a shot — a real threat on goal!",
        "An attempt on goal! Players scramble to react in the box!",
        "The forward drives towards goal and lets fly with a fierce effort!",
        "A dangerous attempt — just wide of the post!",
    ],
    "Side Kick": [
        "A sweeping side-footed pass rolls precisely to the teammate!",
        "Side kick — the player uses the inside of their foot for precision!",
        "Technically superb — a crisp side-foot drive on goal!",
        "A low, side-footed finish makes it look deceptively easy!",
    ],
    "Edge Kick": [
        "A sharp edge-of-the-foot kick curves dangerously into the area!",
        "Stunning technique — the player bends it with the outside of the boot!",
        "An edge-kick delivery whips past the defensive wall!",
        "Incredible curl on that edge kick — the goalkeeper is beaten!",
    ],
    "Penalty Kick": [
        "PENALTY! The crowd erupts — this could be the decisive moment!",
        "Spot kick awarded! The striker steps up for a one-on-one with the goalkeeper!",
        "A penalty kick is given — high drama in the stadium!",
        "The referee points to the spot — a golden opportunity!",
    ],
    "Free Kick": [
        "Free kick in a dangerous position — the specialist lines up!",
        "A whipped free kick swerves over the wall — right into the corner!",
        "The wall is set — the free kick flies towards goal!",
        "Dead ball situation — the team sets up for a direct free kick threat!",
    ],
    "Attacking Transition": [
        "Lightning-fast transition — the team breaks forward at pace!",
        "Brilliant vertical play — the team transitions from defence to attack in seconds!",
        "A sharp attacking move catches the opposition high up the pitch!",
        "Rapid ball movement as the team surges forward in attack!",
    ],
    "Counter Attack": [
        "COUNTER ATTACK! The team explodes forward on the break!",
        "Blistering pace on the counter — defenders scrambling to recover!",
        "Classic counter-attack football — fast, direct, and deadly!",
        "A swift counter catches the opposition out of position!",
    ],
    "Through Pass": [
        "A laser-precise through ball splits the defence wide open!",
        "Visionary passing — the ball is threaded between two defenders!",
        "Outstanding vision! The through pass puts the striker clean through!",
        "A perfectly weighted through ball finds the run in behind!",
    ],
    "Cross": [
        "A teasing cross swings into the danger area — heads up in the box!",
        "The winger whips in a dangerous cross from the flank!",
        "A driven cross into the mixer — who gets on the end of it?",
        "A pinpoint cross finds the run at the far post!",
    ],
    "Crossing Movement": [
        "The winger makes a blistering run and delivers a crossing ball!",
        "Outstanding wide play — the player cuts inside and crosses!",
        "A threatening crossing movement from the left flank!",
        "The wide man creates space and swings in a dangerous cross!",
    ],
    "Header": [
        "A commanding header — the ball is powered towards goal!",
        "The attacker rises high and directs a firm header on target!",
        "Winning the aerial battle — a powerful header from the centre-back!",
        "Superb movement — the striker meets the cross with a glancing header!",
    ],
    "Dribble": [
        "Electrifying dribbling — the winger skips past two defenders effortlessly!",
        "Brilliant ball control takes the player past his marker!",
        "An explosive dribble from the forward — leaving defenders in his wake!",
        "Mesmerising footwork — the player threads through tight space!",
    ],
    "Tackle / Sliding": [
        "A perfectly timed sliding tackle wins the ball cleanly — great defending!",
        "Crucial intervention! The defender throws himself into the tackle!",
        "A crunching but legal challenge dispossesses the attacker!",
        "Last-ditch defending — the slide tackle saves the situation!",
    ],
    "Sliding Interception": [
        "Outstanding read of the game — a sliding interception cuts out the danger!",
        "The defender slides in to intercept before the attacker can react!",
        "Brilliant anticipation — a well-timed sliding interception!",
        "The midfielder slides across to cut out the through ball!",
    ],
    "Defensive Clearance": [
        "A vital defensive clearance — the ball is launched out of the danger zone!",
        "The centre-back clears his lines under heavy pressure!",
        "Relief for the defence — a big clearance from the penalty area!",
        "No hesitation — the defender clears the ball upfield immediately!",
    ],
    "Defending Ball": [
        "Strong defensive positioning — the defender shields the ball excellently!",
        "Smart defending — the player uses their body to block the attacker's path!",
        "Good defensive instinct — tracking the ball and closing space!",
        "The defender stands firm, positioning themselves between player and ball!",
    ],
    "Defensive Interception": [
        "The defence cuts out the pass with a well-timed interception!",
        "Brilliant defensive reading — the pass is intercepted before it reaches its target!",
        "The midfielder steps across to make a crucial interception!",
        "Sharp defensive work — the ball is claimed before the attacker can react!",
    ],
    "Ball Recovery": [
        "Great pressure leads to a ball recovery in a dangerous area!",
        "The team wins the ball back quickly — high press paying dividends!",
        "Excellent work rate — the player chases and recovers possession!",
        "A loose ball is pounced on — quick thinking by the midfielder!",
    ],
    "Aerial Duel": [
        "A towering aerial duel — both players contest the ball in the air!",
        "A mighty battle in the air — muscles and determination on display!",
        "The two players challenge for a header — it's won powerfully!",
        "An intense aerial contest — the stronger jump wins it!",
    ],
    "Aggressive Pressing": [
        "Intense pressing from the team — the opposition can't breathe on the ball!",
        "Relentless high press! Three players close in on the ball carrier!",
        "Aggressive pack pressing forces an error from the opposition!",
        "Suffocating pressure from the team — the opposition is pinned back!",
    ],
    "Corner Kick": [
        "Corner kick awarded — a prime set-piece opportunity!",
        "The ball is swung in from the corner flag — danger in the penalty area!",
        "A corner kick brings all outfield players into the box!",
        "A perfectly executed corner delivery into the six-yard box!",
    ],
    "Throw-in": [
        "A long throw-in hurled into the danger zone — trouble for the defence!",
        "Play restarts with a throw-in near the opposition's penalty area!",
        "The throw-in is used cleverly to maintain possession!",
        "A quick throw-in catches the opposition off-guard!",
    ],
    "Goalkeeper Save": [
        "WHAT A SAVE! The goalkeeper dives full-length — absolutely breathtaking!",
        "Incredible reflexes from the goalkeeper — the team is kept in the game!",
        "Point-blank save! The keeper is the hero — denying the striker completely!",
        "A stunning stop! The goalkeeper gets down low to push the shot wide!",
    ],
    "Foul": [
        "The referee blows for a foul — a reckless challenge from behind!",
        "Foul given! The defender brings down the attacker in full flow!",
        "Illegal challenge! The referee stops play and signals a free kick!",
        "The foul is clumsy and unnecessary — the referee has no hesitation!",
    ],
    "Offside": [
        "Offside! The linesman raises the flag — the goal is disallowed!",
        "Caught offside — the attacker was fractionally ahead of the last defender!",
        "The offside trap is sprung perfectly — the referee's assistant flags!",
        "A close offside decision breaks down what looked a promising attack!",
    ],
    "Yellow / Red Card": [
        "The referee reaches into his pocket — a card is brandished!",
        "Booking! The player receives a yellow card for that reckless challenge!",
        "Disciplinary action! The referee makes an example of the offending player!",
        "A red card is shown! The team is reduced to ten men!",
    ],
    "Pass": [
        "Neat, composed passing as the team builds methodically from the back!",
        "A crisp, accurate pass moves the ball forward at pace!",
        "The midfield is controlling the game through precise short passing!",
        "Quick one-touch passing opens up space on the right flank!",
    ],
    "Celebration": [
        "Wild scenes! Players mob the goalscorer in sheer jubilation!",
        "The team erupts in celebration — pure emotion floods the pitch!",
        "Arms aloft, the players run towards the supporters in delight!",
        "Unforgettable celebrations — this goal clearly means everything!",
    ],
    "Ball in Play": [
        "An intense, open period of play with both teams pressing forward!",
        "The game flows freely — end-to-end football on display!",
        "Both sides battling for possession in an evenly contested spell!",
        "High tempo football — the crowd is on its feet as the action builds!",
    ],
    "Unknown": [
        "A moment of action unfolds on the pitch.",
        "Play continues at intense pace.",
        "An incident occurs as the game develops.",
    ],
}


def generate_commentary(action_label: str, confidence: float) -> str:
    templates = COMMENTARY_TEMPLATES.get(action_label, COMMENTARY_TEMPLATES["Unknown"])
    base = random.choice(templates)
    if confidence < 0.45:
        hedges = ["Possibly —", "It appears that —", "It looks as if —"]
        return f"{random.choice(hedges)} {base.lower()}"
    return base


def generate_full_commentary(results: List[Dict]) -> List[Dict]:
    seen: List[str] = []
    for res in results:
        label = res.get("top_action", "Unknown")
        conf  = res.get("top_confidence", 0.0)
        templates = COMMENTARY_TEMPLATES.get(label, COMMENTARY_TEMPLATES["Unknown"])
        idx = seen.count(label) % len(templates)
        res["commentary"] = templates[idx] if seen.count(label) >= 2 else generate_commentary(label, conf)
        seen.append(label)
    return results


# ─── Professional Match Summary ───────────────────────────────────────────────
SUMMARY_OPENERS = [
    "AI analysis of this football clip reveals",
    "Football intelligence processing identifies",
    "Advanced match analysis detects",
    "Deep tactical analysis highlights",
]

TACTICAL_DESCRIPTORS: Dict[str, str] = {
    "attacking":  ["aggressive attacking gameplay", "high-tempo attacking football",
                   "an intensive attacking build-up", "a sharp attacking transition"],
    "defending":  ["solid defensive organisation", "disciplined defensive pressing",
                   "resilient defensive shape", "determined defensive work"],
    "transition": ["rapid ball transitions", "quick vertical play",
                   "fast switch from defence to attack", "explosive counter-attacking movement"],
    "set_piece":  ["dangerous set-piece situations", "well-worked dead-ball opportunities",
                   "threatening corner and free kick deliveries"],
}

ATTACKING_LABELS = {
    "Goal Scored", "Shot on Target", "Goal Attempt", "Side Kick", "Edge Kick",
    "Penalty Kick", "Free Kick", "Attacking Transition", "Counter Attack",
    "Through Pass", "Cross", "Crossing Movement", "Header", "Dribble",
}
DEFENDING_LABELS = {
    "Tackle / Sliding", "Sliding Interception", "Defensive Clearance",
    "Defending Ball", "Defensive Interception", "Ball Recovery",
    "Aerial Duel", "Aggressive Pressing",
}
SET_PIECE_LABELS = {"Corner Kick", "Throw-in", "Penalty Kick", "Free Kick"}


def generate_match_summary(results: List[Dict]) -> str:
    if not results:
        return "No events detected in this video clip."

    counts    = Counter(r.get("top_action", "Unknown") for r in results)
    all_acts  = set(counts.keys())
    total_s   = results[-1]["end_time"]
    unique_n  = len(counts)

    att_acts  = all_acts & ATTACKING_LABELS
    def_acts  = all_acts & DEFENDING_LABELS
    sp_acts   = all_acts & SET_PIECE_LABELS

    # Find top action with highest confidence
    top_res = max(results, key=lambda x: x.get("top_confidence", 0))
    top_name = top_res["top_action"]
    top_conf = top_res["top_confidence"]

    opener = random.choice(SUMMARY_OPENERS)

    themes: List[str] = []
    if att_acts:
        themes.append(random.choice(TACTICAL_DESCRIPTORS["attacking"]))
    if def_acts:
        themes.append(random.choice(TACTICAL_DESCRIPTORS["defending"]))
    if sp_acts:
        themes.append(random.choice(TACTICAL_DESCRIPTORS["set_piece"]))
    if "Counter Attack" in all_acts or "Attacking Transition" in all_acts:
        themes.append(random.choice(TACTICAL_DESCRIPTORS["transition"]))

    specific: List[str] = []
    for lbl in ["Crossing Movement", "Defensive Interception", "Sliding Interception",
                "Side Kick", "Edge Kick", "Aggressive Pressing", "Ball Recovery"]:
        if lbl in all_acts:
            specific.append(lbl.lower())

    theme_str = ", ".join(themes[:3]) if themes else "competitive match play"
    spec_str  = ""
    if specific:
        spec_str = f" {', '.join(specific[:3]).capitalize()} were also observed during transitional phases of play."

    highlights: List[str] = []
    if "Goal Scored" in all_acts:
        highlights.append("⚽ A goal was scored — the decisive moment of the clip!")
    if "Goalkeeper Save" in all_acts:
        highlights.append("🧤 A key goalkeeper save prevented a goal.")
    if "Penalty Kick" in all_acts:
        highlights.append("🟡 A penalty kick situation was identified.")
    
    # Specific tactical insight mention
    tactical_insight = f"High-confidence {top_name} ({top_conf*100:.0f}%) suggests a key {theme_str} phase."

    summary = (
        f"{opener} {theme_str} across {unique_n} distinct action(s) "
        f"in this {total_s:.0f}-second clip. {tactical_insight}{spec_str}"
    )
    if highlights:
        summary += " | " + " | ".join(highlights)
    return summary


# ─── Match Analysis Card ─────────────────────────────────────────────────────
def generate_match_analysis_card(results: List[Dict]) -> Dict:
    """
    Returns a structured analysis card dict for the UI to render.

    Keys: style, detected_themes, tactical_notes, overall_rating
    """
    counts   = Counter(r.get("top_action", "Unknown") for r in results)
    all_acts = set(counts.keys())

    att_count = len(all_acts & ATTACKING_LABELS)
    def_count = len(all_acts & DEFENDING_LABELS)
    sp_count  = len(all_acts & SET_PIECE_LABELS)

    # Determine overall style
    if att_count > def_count + 1:
        style = "High-Tempo Attacking Football"
    elif def_count > att_count + 1:
        style = "Disciplined Defensive Organisation"
    elif sp_count >= 2:
        style = "Set-Piece Focused Play"
    else:
        style = "Balanced Competitive Football"

    # Detected themes (checkmarks)
    theme_checks: Dict[str, bool] = {
        "Attacking Build-up":        len(all_acts & {"Pass", "Through Pass", "Attacking Transition"}) > 0,
        "Crossing Opportunity":      len(all_acts & {"Cross", "Crossing Movement"}) > 0,
        "Defensive Pressing":        len(all_acts & {"Aggressive Pressing", "Defending Ball"}) > 0,
        "Ball Recovery":             "Ball Recovery" in all_acts,
        "Goal Attempt":              len(all_acts & {"Goal Scored", "Shot on Target", "Goal Attempt"}) > 0,
        "Sliding Tackle":            len(all_acts & {"Tackle / Sliding", "Sliding Interception"}) > 0,
        "Counter Attack":            len(all_acts & {"Counter Attack", "Attacking Transition"}) > 0,
        "Set-Piece Threat":          sp_count > 0,
        "Aerial Battles":            "Aerial Duel" in all_acts or "Header" in all_acts,
        "Technical Skill (Dribble)": "Dribble" in all_acts,
        "Edge / Side Kick":          len(all_acts & {"Side Kick", "Edge Kick"}) > 0,
        "Defensive Interception":    "Defensive Interception" in all_acts,
        "Keeper Engagement":         "Goalkeeper Save" in all_acts,
    }

    # Tactical notes
    notes: List[str] = []
    if "Aggressive Pressing" in all_acts:
        notes.append("High press was used effectively to win back possession.")
    if "Counter Attack" in all_acts:
        notes.append("Fast counter-attacks exposed space behind the defensive line.")
    if len(all_acts & {"Side Kick", "Edge Kick"}) > 0:
        notes.append("Technical kicking variations (side/edge) were employed.")
    if "Goalkeeper Save" in all_acts:
        notes.append("The goalkeeper played a crucial role in preventing goals.")
    if "Corner Kick" in all_acts or "Free Kick" in all_acts:
        notes.append("Set pieces provided dangerous goal-scoring opportunities.")

    # Overall rating
    high_conf_count = sum(1 for r in results if r.get("top_confidence", 0) > 0.70)
    total           = len(results)
    rating_pct      = int((high_conf_count / total * 100)) if total else 50

    return {
        "style":           style,
        "theme_checks":    theme_checks,
        "tactical_notes":  notes,
        "overall_rating":  rating_pct,
    }
