"""Queue decision rules and a stable JSON/OpenCV annotation contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DecisionThresholds:
    """Operational thresholds; tune these for the store and camera view."""

    queue_warning: int = 4
    queue_critical: int = 8
    wait_warning_sec: float = 60.0
    wait_critical_sec: float = 120.0

    def __post_init__(self) -> None:
        if self.queue_warning < 1 or self.queue_critical < self.queue_warning:
            raise ValueError("queue thresholds must satisfy 1 <= warning <= critical")
        if self.wait_warning_sec <= 0 or self.wait_critical_sec < self.wait_warning_sec:
            raise ValueError("wait thresholds must satisfy 0 < warning <= critical")


def evaluate_queue(metrics: dict[str, Any], thresholds: DecisionThresholds,
                   frame_index: int) -> dict[str, Any]:
    """Apply deterministic rules and return the Day 4 JSON contract."""
    queue_length = max(0, int(metrics.get("queue_length", 0)))
    avg_wait_sec = max(0.0, float(metrics.get("avg_wait_sec", 0.0)))
    alerts = list(metrics.get("alerts", []))
    rule_hits: list[str] = []

    if queue_length >= thresholds.queue_critical:
        rule_hits.append("queue_critical")
        alerts.append({"type": "queue_threshold", "severity": "critical",
                       "queue_length": queue_length})
    elif queue_length >= thresholds.queue_warning:
        rule_hits.append("queue_warning")
        alerts.append({"type": "queue_threshold", "severity": "warning",
                       "queue_length": queue_length})

    if avg_wait_sec >= thresholds.wait_critical_sec:
        rule_hits.append("wait_critical")
        alerts.append({"type": "average_wait_threshold", "severity": "critical",
                       "avg_wait_sec": round(avg_wait_sec, 2)})
    elif avg_wait_sec >= thresholds.wait_warning_sec:
        rule_hits.append("wait_warning")
        alerts.append({"type": "average_wait_threshold", "severity": "warning",
                       "avg_wait_sec": round(avg_wait_sec, 2)})

    if any(hit.endswith("critical") for hit in rule_hits):
        status = "critical"
        recommendation = "Open another checkout lane and call for staff support."
    elif rule_hits:
        status = "warning"
        recommendation = "Monitor the queue and prepare checkout support."
    else:
        status = "normal"
        recommendation = "Queue is within the configured thresholds."

    return {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame_index": int(frame_index),
        "queue_length": queue_length,
        "avg_wait_sec": round(avg_wait_sec, 2),
        "status": status,
        "recommendation": recommendation,
        "rule_hits": rule_hits,
        "alerts": alerts,
    }


def annotate_frame(frame: np.ndarray, polygon: np.ndarray,
                   result: dict[str, Any],
                   person_boxes: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Draw ROI and a readable decision banner on a copy of the source frame."""
    annotated = frame.copy()
    points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    status = result["status"]
    color = {"normal": (50, 190, 70), "warning": (0, 180, 255),
             "critical": (30, 30, 230)}.get(status, (180, 180, 180))
    cv2.polylines(annotated, [points], True, (0, 220, 255), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in person_boxes or []:
        foot = (int((x1 + x2) / 2), int(y2))
        inside = cv2.pointPolygonTest(points, (float(foot[0]), float(foot[1])), False) >= 0
        box_color = (60, 220, 60) if inside else (160, 160, 160)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(annotated, foot, 4, (0, 0, 255), -1)
        cv2.putText(annotated, "in queue" if inside else "person",
                    (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, box_color, 1, cv2.LINE_AA)

    height, width = annotated.shape[:2]
    banner_h = min(92, max(62, height // 5))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (18, 30, 48), -1)
    annotated = cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0)
    scale = max(0.45, min(0.8, width / 1000.0))
    thickness = max(1, int(round(scale * 2)))
    cv2.putText(annotated,
                f"Queue: {result['queue_length']}  Avg wait: {result['avg_wait_sec']:.1f}s  Status: {status.upper()}",
                (12, max(24, int(31 * scale / 0.7))), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    short_action = result["recommendation"][:100]
    cv2.putText(annotated, short_action, (12, min(banner_h - 8, max(48, int(65 * scale / 0.7)))),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.38, scale * 0.75), (245, 245, 245),
                1, cv2.LINE_AA)
    return annotated
