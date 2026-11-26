# Peopletracking

This project has a webcam people counter and a camera-based 2D LiDAR simulator.
Both run locally with OpenCV and Ultralytics YOLO. See the
[LiDAR setup guide](LiDAR-setup.md) for floor calibration and scan logging.

## Webcam people counter

From this folder, create an isolated environment and start the counter.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python people_counter.py
```

Use Python 3.12 if your installed Python version lacks compatible packages. The
first run with a working camera downloads the YOLO11 nano weights. If macOS
prompts for camera access, allow it for the app launching Python. Use
`--camera 1` to select a second camera.

Keep the camera fixed. Walk across the yellow center line with your torso visible.
Use the direction shown in the preview. Left to right adds one and right to left
subtracts one. The count starts at zero and can go negative. The center of a
person's box must clear a small margin on each side before a crossing counts.

Press Q or Esc in the preview to quit. Frames are processed locally.
