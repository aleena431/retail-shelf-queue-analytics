"""Checkout queue dwell-time monitor using Ultralytics tracking and polygon ROIs.

Example:
    python checkout_queue_monitor.py --source checkout.mp4 \
        --roi 100,100 500,100 520,400 80,400

The latest result is emitted as JSON with queue_length, avg_wait_sec, and alerts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    entered_at: float
    last_seen_at: float
    dwell_sec: float = 0.0


@dataclass
class QueueDwellMonitor:
    """Maintain per-track dwell time while a person's foot point is in an ROI."""

    polygon: np.ndarray
    alert_after_sec: float = 60.0
    stale_after_sec: float = 3.0
    tracks: dict[int, TrackState] = field(default_factory=dict)
    _alerted: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        points = np.asarray(self.polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("polygon must contain at least three (x, y) points")
        self.polygon = points
        if self.alert_after_sec <= 0 or self.stale_after_sec < 0:
            raise ValueError("alert_after_sec must be positive and stale_after_sec nonnegative")

    def update(self, people: dict[int, tuple[int, int]], now: float | None = None) -> dict[str, Any]:
        """Update with {track_id: (foot_x, foot_y)} detections for one frame."""
        now = time.monotonic() if now is None else now
        inside_ids: set[int] = set()
        for track_id, (x, y) in people.items():
            inside = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0
            state = self.tracks.get(track_id)
            if inside:
                inside_ids.add(track_id)
                if state is None:
                    state = TrackState(entered_at=now, last_seen_at=now)
                    self.tracks[track_id] = state
                else:
                    # Accumulate only while seen inside. Brief detector gaps are
                    # tolerated up to stale_after_sec without extending dwell.
                    elapsed = max(0.0, now - state.last_seen_at)
                    state.dwell_sec += min(elapsed, self.stale_after_sec)
                    state.last_seen_at = now

        # Remove departed IDs after their last in-zone observation. When they
        # return later they begin a new queue visit.
        for track_id, state in list(self.tracks.items()):
            if track_id not in inside_ids and now - state.last_seen_at > self.stale_after_sec:
                del self.tracks[track_id]
                self._alerted.discard(track_id)

        active = [(tid, st) for tid, st in self.tracks.items()
                  if now - st.last_seen_at <= self.stale_after_sec]
        # Include time from last observation to this frame for currently seen IDs.
        waits = [st.dwell_sec + max(0.0, now - st.last_seen_at)
                 for tid, st in active if tid in inside_ids]
        alerts = []
        for tid, wait in zip([tid for tid, _ in active if tid in inside_ids], waits):
            if wait >= self.alert_after_sec and tid not in self._alerted:
                alerts.append({"type": "long_wait", "track_id": tid,
                               "wait_sec": round(wait, 1)})
                self._alerted.add(tid)
        return {
            "queue_length": len(inside_ids),
            "avg_wait_sec": round(sum(waits) / len(waits), 2) if waits else 0.0,
            "alerts": alerts,
        }


def _parse_roi(values: list[str]) -> np.ndarray:
    try:
        pts = [tuple(map(int, value.split(","))) for value in values]
        return np.asarray(pts, dtype=np.int32)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("ROI points must look like x,y") from exc


def run(source: str, roi: np.ndarray, model_path: str, alert_after_sec: float,
        show: bool = False) -> None:
    model = YOLO(model_path)
    monitor = QueueDwellMonitor(roi, alert_after_sec=alert_after_sec)
    cap = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 classes=[0], verbose=False)[0]
            people: dict[int, tuple[int, int]] = {}
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                    people[track_id] = (int((x1 + x2) / 2), int(y2))
            payload = monitor.update(people)
            print(json.dumps(payload), flush=True)
            if show:
                cv2.polylines(frame, [roi], True, (0, 220, 255), 2)
                cv2.putText(frame, f"Queue: {payload['queue_length']}  Avg: {payload['avg_wait_sec']:.1f}s",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 240, 30), 2)
                cv2.imshow("Checkout Queue Dwell Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Track checkout queue dwell time in a polygon ROI")
    parser.add_argument("--source", default="0", help="Video path or camera index (default: 0)")
    parser.add_argument("--roi", nargs="+", required=True, metavar="X,Y",
                        help="At least three polygon vertices, e.g. 100,100 500,100 500,400")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--alert-after", type=float, default=60.0,
                        help="Emit one long_wait alert per visit after this many seconds")
    parser.add_argument("--show", action="store_true", help="Display annotated preview (press q to quit)")
    args = parser.parse_args()
    run(args.source, _parse_roi(args.roi), args.model, args.alert_after, args.show)


if __name__ == "__main__":
    main()
