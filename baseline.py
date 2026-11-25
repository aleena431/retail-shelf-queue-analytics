"""FastAPI baseline for CPU person detection on a still image.

Run the PowerShell command shown in README.md.
"""

from __future__ import annotations

import base64
import binascii
import os
import threading
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11n.pt")
MAX_IMAGE_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="Retail Shelf & Queue Analytics Baseline",
    description="CPU-friendly baseline health and person-detection API.",
    version="0.1.0",
)
_model: YOLO | None = None
_model_lock = threading.Lock()


class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


def get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service readiness without loading model weights."""
    return {
        "status": "ok",
        "service": "retail-queue-baseline",
        "model": MODEL_PATH,
        "model_loaded": _model is not None,
        "device": "cpu",
    }


@app.post("/detect")
def detect(payload: DetectRequest) -> dict[str, Any]:
    """Detect COCO person class (0) in one base64-encoded image."""
    try:
        image_bytes = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 must be valid base64") from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="image_base64 is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the 15 MB limit")

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="decoded data is not a supported image")

    model = get_model()
    result = model.predict(
        source=image,
        classes=[0],
        conf=payload.confidence,
        device="cpu",
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append({
                "class_id": class_id,
                "class_name": str(model.names[class_id]),
                "confidence": round(float(score), 4),
                "bbox_xyxy": [round(float(value), 1) for value in box],
            })

    return {"person_count": len(detections), "detections": detections}


