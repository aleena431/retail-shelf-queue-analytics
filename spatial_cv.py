"""Spatial overlays for retail people analytics.

Processes one image or a video with YOLO person detections, then renders a
polygon ROI, line-crossing counter, occupancy grid, face blur, and dwell-density
heatmap. Video mode uses ByteTrack IDs; image mode demonstrates spatial logic
for a single frame only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_normalized_polygon(values: list[str] | None) -> list[tuple[float, float]]:
    if not values:
        return [(0.05, 0.08), (0.95, 0.08), (0.95, 0.99), (0.05, 0.99)]
    try:
        points = [tuple(float(v) for v in item.split(",")) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI points must use normalized x,y values such as 0.1,0.2") from exc
    if any(len(point) != 2 or not all(0.0 <= v <= 1.0 for v in point) for point in points):
        raise argparse.ArgumentTypeError("ROI coordinates must be x,y values between 0 and 1")
    if len(points) < 3:
        raise argparse.ArgumentTypeError("ROI needs at least three vertices")
    return points


class SpatialOverlayProcessor:
    def __init__(self, model_path: str, rows: int, cols: int,
                 roi_normalized: list[tuple[float, float]], line_y: float,
                 blur_faces: bool = True) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("grid rows and columns must be positive")
        if not 0.0 < line_y < 1.0:
            raise ValueError("line-y must be between 0 and 1")
        self.model = YOLO(model_path)
        self.rows = rows
        self.cols = cols
        self.roi_normalized = roi_normalized
        self.line_y = line_y
        self.blur_faces = blur_faces
        self.previous_line_side: dict[int, int] = {}
        self.crossing_events: list[dict[str, Any]] = []
        self.heatmap: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.last_grid_counts: list[list[int]] = []
        self.last_roi_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.head_blur_count = 0

    def process(self, frame: np.ndarray, track: bool) -> np.ndarray:
        height, width = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.heatmap *= 0.97
        polygon = np.asarray(
            [(round(x * (width - 1)), round(y * (height - 1)))
             for x, y in self.roi_normalized], dtype=np.int32
        )
        line_y_px = round(self.line_y * (height - 1))

        if track:
            result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                      classes=[0], device="cpu", verbose=False)[0]
        else:
            result = self.model.predict(frame, classes=[0], device="cpu", verbose=False)[0]

        annotated = frame.copy()

        detections: list[tuple[int | None, tuple[int, int, int, int], tuple[int, int], bool]] = []
        boxes = result.boxes
        if boxes is not None:
            coords = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().tolist()
            ids = boxes.id.int().cpu().tolist() if track and boxes.id is not None else [None] * len(coords)
            counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            roi_count = 0
            for (x1, y1, x2, y2), confidence, track_id in zip(coords, scores, ids):
                ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
                if self.blur_faces:
                    # Blur the head/face area estimated from the top of the
                    # person box; this works without a separate face model.
                    px1, px2 = max(0, ix1), min(width, ix2)
                    py1, py2 = max(0, iy1), min(height, iy1 + max(1, int((iy2 - iy1) * 0.30)))
                    region = annotated[py1:py2, px1:px2]
                    if region.size and min(region.shape[:2]) >= 3:
                        kernel = min(31, max(3, (min(region.shape[:2]) // 2) | 1))
                        annotated[py1:py2, px1:px2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
                        self.head_blur_count += 1
                foot = (int((x1 + x2) / 2), int(y2))
                inside = cv2.pointPolygonTest(polygon, (float(foot[0]), float(foot[1])), False) >= 0
                detections.append((track_id, (ix1, iy1, ix2, iy2), foot, inside))
                self.total_detections += 1
                if inside:
                    roi_count += 1
                    cell_col = min(self.cols - 1, max(0, foot[0] * self.cols // width))
                    cell_row = min(self.rows - 1, max(0, foot[1] * self.rows // height))
                    counts[cell_row][cell_col] += 1
                    cv2.circle(self.heatmap, foot, 24, 1.0, thickness=-1)
                if track_id is not None:
                    side = 1 if foot[1] >= line_y_px else -1
                    old_side = self.previous_line_side.get(track_id)
                    if old_side is not None and side != old_side:
                        direction = "down" if side > old_side else "up"
                        self.crossing_events.append({"track_id": track_id, "direction": direction,
                                                     "frame": self.frame_count})
                    self.previous_line_side[track_id] = side
            self.last_grid_counts = counts
            self.last_roi_count = roi_count
        else:
            self.last_grid_counts = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            self.last_roi_count = 0

        # Tint occupied grid cells, then draw cell boundaries and counts.
        tint = annotated.copy()
        cell_w = width / self.cols
        cell_h = height / self.rows
        for row in range(self.rows):
            for col in range(self.cols):
                if self.last_grid_counts[row][col] > 0:
                    x1, y1 = int(col * cell_w), int(row * cell_h)
                    x2, y2 = int((col + 1) * cell_w), int((row + 1) * cell_h)
                    cv2.rectangle(tint, (x1, y1), (x2, y2), (255, 150, 30), -1)
        annotated = cv2.addWeighted(tint, 0.16, annotated, 0.84, 0)
        for row in range(1, self.rows):
            y = round(row * cell_h)
            cv2.line(annotated, (0, y), (width - 1, y), (235, 235, 235), 1)
        for col in range(1, self.cols):
            x = round(col * cell_w)
            cv2.line(annotated, (x, 0), (x, height - 1), (235, 235, 235), 1)
        for row in range(self.rows):
            for col in range(self.cols):
                x, y = int(col * cell_w + 7), int(row * cell_h + 18)
                cv2.putText(annotated, str(self.last_grid_counts[row][col]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, str(self.last_grid_counts[row][col]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 40, 60), 1, cv2.LINE_AA)

        cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 220, 255), thickness=2)
        cv2.line(annotated, (0, line_y_px), (width - 1, line_y_px), (255, 80, 180), 2)
        for track_id, (x1, y1, x2, y2), foot, inside in detections:
            color = (60, 220, 60) if inside else (160, 160, 160)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, foot, 4, (0, 0, 255), -1)
            label = f"person {track_id}" if track_id is not None else "person"
            cv2.putText(annotated, label, (x1, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        cv2.putText(annotated, f"ROI: {self.last_roi_count}  Crossings: {len(self.crossing_events)}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 240, 40), 2, cv2.LINE_AA)
        self.last_frame = annotated.copy()
        self.frame_count += 1
        return annotated

    def save_heatmap(self, path: Path) -> None:
        if self.heatmap is None or self.last_frame is None:
            raise RuntimeError("No frames were processed")
        smooth = cv2.GaussianBlur(self.heatmap, (0, 0), sigmaX=22, sigmaY=22)
        if float(smooth.max()) <= 0:
            heat = np.zeros_like(self.last_frame)
        else:
            scaled = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            mask = (scaled > 8).astype(np.uint8)[:, :, None]
            blended = cv2.addWeighted(self.last_frame, 0.55, colored, 0.45, 0)
            heat = np.where(mask > 0, blended, self.last_frame)
        if not cv2.imwrite(str(path), heat):
            raise RuntimeError(f"Could not write heatmap: {path}")

    def metrics(self) -> dict[str, Any]:
        return {
            "frames_processed": self.frame_count,
            "person_detections_total": self.total_detections,
            "people_in_roi_last_frame": self.last_roi_count,
            "line_crossings": self.crossing_events,
            "grid_occupancy_last_frame": self.last_grid_counts,
            "head_regions_blurred_total": self.head_blur_count,
        }


def process_source(source: str, output_dir: Path, model_path: str, rows: int, cols: int,
                   roi: list[tuple[float, float]], line_y: float,
                   blur_faces: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source)
    is_camera = source.isdecimal()
    is_image = not is_camera and source_path.suffix.lower() in IMAGE_SUFFIXES
    processor = SpatialOverlayProcessor(model_path, rows, cols, roi, line_y, blur_faces)

    if is_image:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        overlay = processor.process(frame, track=False)
        overlay_path = output_dir / f"overlay_{source_path.stem}.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Could not write overlay: {overlay_path}")
    else:
        video_source: int | str = int(source) if is_camera else source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        overlay_path = output_dir / f"overlay_{source_path.stem or 'camera'}.mp4"
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create overlay video: {overlay_path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(processor.process(frame, track=True))
        finally:
            cap.release()
            writer.release()

    heatmap_path = output_dir / "heatmap.png"
    processor.save_heatmap(heatmap_path)
    metrics = processor.metrics()
    metrics.update({"source": source, "overlay": str(overlay_path), "heatmap": str(heatmap_path)})
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Create polygon, line, grid, blur, and heatmap overlays")
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument("--output-dir", default="output/spatial", help="Directory for overlay, heatmap, and metrics")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model weights")
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--roi", nargs="+", metavar="X,Y",
                        help="Normalized polygon vertices (0 to 1), e.g. 0.1,0.2 0.9,0.2 0.9,0.9")
    parser.add_argument("--line-y", type=float, default=0.65, help="Normalized horizontal line position (0 to 1)")
    parser.add_argument("--no-head-blur", action="store_true", help="Disable blur over detected person head regions")
    args = parser.parse_args()
    roi = parse_normalized_polygon(args.roi)
    metrics = process_source(args.source, Path(args.output_dir), args.model,
                             args.grid_rows, args.grid_cols, roi,
                             args.line_y, blur_faces=not args.no_head_blur)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
