# Checkout Queue Dwell-Time Monitor

CPU-friendly checkout queue monitor using Ultralytics YOLO11n (or YOLOv8n), ByteTrack, OpenCV polygon zones, and Python monotonic time. It includes a FastAPI baseline for still-image person detection.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The virtual environment can be used directly; PowerShell activation is optional.

## Day 2 baseline API

Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn baseline:app --host 127.0.0.1 --port 8000
```

Check service health in another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The health route returns `status`, configured model, whether the model has been loaded, and the CPU device. To detect people in one image, open `http://127.0.0.1:8000/docs`, choose `POST /detect`, and send JSON with a base64-encoded JPEG or PNG in `image_base64`. The model loads on the first `/detect` request, and inference runs on CPU. The endpoint returns `person_count` and bounding boxes. This still-image baseline does not produce track IDs or dwell times.

## Queue monitor

Run against a camera or video. Set polygon vertices in the source frame's pixel coordinates:

```powershell
.\.venv\Scripts\python.exe checkout_queue_monitor.py --source 0 --roi 100,100 500,100 520,400 80,400 --show
```

For a video file, replace `0` with its path. The default model is `yolo11n.pt`; use `--model yolov8n.pt` to choose YOLOv8n. Ultralytics downloads weights on first use. Press `q` to close the preview. The yellow polygon should cover the floor where people wait, including their feet.

Each frame prints JSON to stdout:

```json
{"queue_length": 2, "avg_wait_sec": 14.25, "alerts": []}
```

`queue_length` counts detected tracks whose bottom-center point falls inside the polygon. `avg_wait_sec` is the mean current dwell time for those tracks. A `long_wait` alert is emitted once per visit after 60 seconds by default; change this with `--alert-after 90`.

## Run without video or camera

```powershell
.\.venv\Scripts\python.exe demo_simulated_queue.py
```

This prints sample queue snapshots and a long-wait alert using synthetic track IDs and timestamps. It demonstrates the dwell-time logic only; it does not represent real tracking output.

## Samples and dataset

Place permitted retail sample images or clips in `samples/`. No media is included in the repository yet. A local Roboflow YOLOv8 dataset with 666 labeled still images of people can be used for person-detection experiments; the dataset metadata identifies CC BY 4.0. Still images do not provide temporal data for validating ByteTrack or real wait times.

## Day 3 spatial overlays

Run the spatial CV script on a still image or video. The defaults draw a normalized polygon ROI, a horizontal line, a 3x4 occupancy grid, blur over detected person head regions, and a density heatmap. Video input uses ByteTrack; still-image input shows one-frame geometry and occupancy only.

```powershell
$img = Get-ChildItem -LiteralPath "C:\Users\ZAEEM\Downloads\queue.v1i.yolov8\train\images" -Filter "*.jpg" | Select-Object -First 1
.\.venv\Scripts\python.exe spatial_cv.py --source $img.FullName --output-dir output/spatial
```

The command writes `overlay_<image-name>.jpg`, `heatmap.png`, and `metrics.json` under `output/spatial`. The output folder is ignored by Git. Customize the zone and grid, for example:

```powershell
.\.venv\Scripts\python.exe spatial_cv.py --source path\to\clip.mp4 --roi 0.1,0.2 0.9,0.2 0.9,0.9 0.1,0.9 --grid-rows 4 --grid-cols 5 --line-y 0.6
```

