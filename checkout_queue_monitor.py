"""Checkout queue dwell-time monitor with decision rules and JSON output.

Example:
    python checkout_queue_monitor.py --source checkout.mp4 \
        --roi 100,100 500,100 520,400 80,400 --show

Outputs per-frame JSON to stdout and metrics.jsonl, the final metrics.json,
and annotated video/frame files under output/day4 by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from decision_rules import DecisionThresholds, annotate_frame, evaluate_queue


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
                    elapsed = max(0.0, now - state.last_seen_at)
                    state.dwell_sec += min(elapsed, self.stale_after_sec)
                    state.last_seen_at = now

        for track_id, state in list(self.tracks.items()):
            if track_id not in inside_ids and now - state.last_seen_at > self.stale_after_sec:
                del self.tracks[track_id]
                self._alerted.discard(track_id)

        active = [(tid, st) for tid, st in self.tracks.items()
                  if now - st.last_seen_at <= self.stale_after_sec]
        waits = [st.dwell_sec + max(0.0, now - st.last_seen_at)
                 for tid, st in active if tid in inside_ids]
        alerts = []
        active_inside_ids = [tid for tid, _ in active if tid in inside_ids]
        for tid, wait in zip(active_inside_ids, waits):
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
        show: bool = False, output_dir: Path = Path("output/day4"),
        thresholds: DecisionThresholds | None = None) -> dict[str, Any]:
    thresholds = thresholds or DecisionThresholds()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    monitor = QueueDwellMonitor(roi, alert_after_sec=alert_after_sec)
    input_path = Path(source)
    is_image = input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_camera = source.isdecimal()
    cap: cv2.VideoCapture | None = None
    image: np.ndarray | None = None
    if is_image:
        image = cv2.imread(str(input_path))
        if image is None:
            raise RuntimeError(f"Could not read image source: {source}")
    else:
        cap = cv2.VideoCapture(int(source) if is_camera else source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0.0
    fps = fps if fps and fps > 0 else 25.0
    video_path = None if is_image else output_dir / (
        "annotated_camera.mp4" if is_camera else f"annotated_{input_path.stem}.mp4"
    )
    latest_frame_path = output_dir / "latest_annotated.jpg"
    jsonl_path = output_dir / "metrics.jsonl"
    final_json_path = output_dir / "metrics.json"
    writer: cv2.VideoWriter | None = None
    latest: dict[str, Any] | None = None
    frame_index = 0

    try:
        with jsonl_path.open("w", encoding="utf-8") as frame_log:
            while True:
                if is_image:
                    if frame_index > 0:
                        break
                    frame = image.copy()
                    result = model.predict(frame, classes=[0], device="cpu", verbose=False)[0]
                    now = 0.0
                else:
                    assert cap is not None
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not create annotated video: {video_path}")
                    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                         classes=[0], device="cpu", verbose=False)[0]
                    # Use media time for clips so CPU inference speed does not inflate dwell.
                    now = time.monotonic() if is_camera else frame_index / fps

                people: dict[int, tuple[int, int]] = {}
                detection_boxes: list[tuple[int, int, int, int]] = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    detection_boxes = [tuple(map(int, box)) for box in xyxy]
                    if not is_image and boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                    else:
                        # A still frame has no persistent IDs; these temporary IDs
                        # support in-zone counting only and are discarded immediately.
                        ids = [-(index + 1) for index in range(len(xyxy))]
                    for track_id, (x1, y1, x2, y2) in zip(ids, xyxy):
                        people[track_id] = (int((x1 + x2) / 2), int(y2))

                queue_metrics = monitor.update(people, now=now)
                latest = evaluate_queue(queue_metrics, thresholds, frame_index)
                annotated = annotate_frame(frame, roi, latest, detection_boxes)
                latest["annotated_frame"] = str(latest_frame_path)
                latest["annotated_video"] = str(video_path) if video_path else None
                if writer is not None:
                    writer.write(annotated)
                if not cv2.imwrite(str(latest_frame_path), annotated):
                    raise RuntimeError(f"Could not write annotated frame: {latest_frame_path}")
                frame_log.write(json.dumps(latest, separators=(",", ":")) + "\n")
                frame_log.flush()
                print(json.dumps(latest), flush=True)
                frame_index += 1
                if show:
                    cv2.imshow("Checkout Queue Decision Monitor", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if latest is None:
        raise RuntimeError(f"No frames were read from source: {source}")
    latest["frames_processed"] = frame_index
    latest["source"] = source
    latest["thresholds"] = {
        "queue_warning": thresholds.queue_warning,
        "queue_critical": thresholds.queue_critical,
        "wait_warning_sec": thresholds.wait_warning_sec,
        "wait_critical_sec": thresholds.wait_critical_sec,
    }
    final_json_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a checkout queue and emit Day 4 decisions and annotations")
    parser.add_argument("--source", default="0", help="Video path or camera index (default: 0)")
    parser.add_argument("--roi", nargs="+", required=True, metavar="X,Y",
                        help="At least three polygon vertices, e.g. 100,100 500,100 500,400")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    parser.add_argument("--alert-after", type=float, default=60.0,
                        help="Emit one long_wait alert per visit after this many seconds")
    parser.add_argument("--queue-warning", type=int, default=4, help="Queue warning threshold")
    parser.add_argument("--queue-critical", type=int, default=8, help="Queue critical threshold")
    parser.add_argument("--wait-warning", type=float, default=60.0, help="Average wait warning in seconds")
    parser.add_argument("--wait-critical", type=float, default=120.0, help="Average wait critical in seconds")
    parser.add_argument("--output-dir", default="output/day4", help="Directory for JSON and annotated outputs")
    parser.add_argument("--show", action="store_true", help="Display annotated preview (press q to stop)")
    args = parser.parse_args()
    thresholds = DecisionThresholds(args.queue_warning, args.queue_critical,
                                    args.wait_warning, args.wait_critical)
    result = run(args.source, _parse_roi(args.roi), args.model, args.alert_after,
                 args.show, Path(args.output_dir), thresholds)
    print(f"Final metrics written to {Path(args.output_dir) / 'metrics.json'}")
    print(f"Annotated frame written to {result['annotated_frame']}")
    if result['annotated_video']:
        print(f"Annotated video written to {result['annotated_video']}")


if __name__ == "__main__":
    main()
