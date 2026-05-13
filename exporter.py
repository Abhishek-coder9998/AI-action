"""
exporter.py
-----------
Export analysis results to JSON / CSV formats.
"""

import json
import csv
import io
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def results_to_json(results: List[Dict], summary: str = "", video_info: Dict = None) -> str:
    """
    Serialize full analysis results to a formatted JSON string.

    Parameters
    ----------
    results    : annotated segment dicts (with commentary added)
    summary    : match summary text
    video_info : metadata dict from VideoProcessor.get_info()

    Returns
    -------
    str : Pretty-printed JSON.
    """
    payload: Dict[str, Any] = {
        "video_info": video_info or {},
        "summary":    summary,
        "events":     [],
    }

    for seg in results:
        payload["events"].append({
            "segment":    seg.get("segment_id", 0),
            "time_start": seg.get("start_time", 0),
            "time_end":   seg.get("end_time", 0),
            "timestamp":  _fmt_ts(seg.get("start_time", 0)),
            "action":     seg.get("top_action", "Unknown"),
            "confidence": round(seg.get("top_confidence", 0.0) * 100, 1),
            "commentary": seg.get("commentary", ""),
            "top_6_actions": [
                {
                    "action":     a.get("label", ""),
                    "confidence": round(a.get("confidence", 0.0) * 100, 1),
                }
                for a in seg.get("top_k_actions", [])
            ],
        })

    return json.dumps(payload, indent=2, ensure_ascii=False)


def results_to_csv(results: List[Dict]) -> str:
    """
    Serialize results to CSV string (for st.download_button).

    Returns
    -------
    str : CSV content as string.
    """
    output = io.StringIO()
    fieldnames = [
        "Segment", "Start (s)", "End (s)", "Timestamp",
        "Action", "Confidence (%)", "Commentary",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for seg in results:
        writer.writerow({
            "Segment":         seg.get("segment_id", 0),
            "Start (s)":       seg.get("start_time", 0),
            "End (s)":         seg.get("end_time", 0),
            "Timestamp":       _fmt_ts(seg.get("start_time", 0)),
            "Action":          seg.get("top_action", "Unknown"),
            "Confidence (%)":  round(seg.get("top_confidence", 0.0) * 100, 1),
            "Commentary":      seg.get("commentary", ""),
        })

    return output.getvalue()


def build_timeline_text(results: List[Dict]) -> str:
    """
    Return a plain-text event timeline:

      00:00 - Ball in Play
      00:15 - Pass
      ...
    """
    lines = []
    for seg in results:
        ts    = _fmt_ts(seg.get("start_time", 0))
        label = seg.get("top_action", "Unknown")
        conf  = seg.get("top_confidence", 0.0)
        lines.append(f"{ts}  –  {label}  ({conf * 100:.0f}%)")
    return "\n".join(lines)


def _fmt_ts(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
