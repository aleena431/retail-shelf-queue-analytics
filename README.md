# Checkout Queue Dwell-Time Monitor

CPU-friendly checkout queue monitor using Ultralytics YOLO, ByteTrack, OpenCV polygon zones, and Python monotonic time.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The virtual environment can be used directly; PowerShell activation is optional. Run a camera or video source with a polygon ROI in source-frame pixel coordinates:

```powershell
.\.venv\Scripts\python.exe checkout_queue_monitor.py --source 0 --roi 100,100 500,100 520,400 80,400 --show
```

For a video file, replace `0` with its path, such as `samples/checkout.mp4`. The default model is `yolo11n.pt`; use `--model yolov8n.pt` to choose YOLOv8n. Ultralytics downloads weights on first run. Press `q` to close the preview. The preview's yellow polygon should cover the floor area where people wait, including their feet.

Each frame prints JSON to stdout:

```json
{"queue_length": 2, "avg_wait_sec": 14.25, "alerts": []}
```

`queue_length` counts detected tracks whose bottom-center point falls inside the polygon. `avg_wait_sec` is the mean current dwell time for those tracks. A `long_wait` alert is emitted once per visit after 60 seconds by default; change this with `--alert-after 90`.

## Samples

Place consented retail sample images or short clips in `samples/`. See `samples/README.md`. No sample media was supplied for this initial commit.
## Run without video or camera

To demonstrate the queue and dwell-time logic offline with simulated track IDs and timestamps:

```powershell
.\.venv\Scripts\python.exe demo_simulated_queue.py
```

This prints sample JSON snapshots and a long-wait alert. It verifies the metric logic only; simulated IDs are not YOLO detections or ByteTrack results. For real tracking validation, use a video or live camera.

