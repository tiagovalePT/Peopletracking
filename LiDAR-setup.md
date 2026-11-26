# Camera-based 2D LiDAR simulator

Quit the original People Counter preview before starting the simulator so the DJI
camera is free. Mark a rectangle on the floor with tape and measure its width and
depth in metres. Position the camera so it sees all four markers and people’s feet.

From Peopletracking, run the following, substituting your measured dimensions.

```sh
.venv/bin/python lidar_counter.py --camera 0 --width 3 --depth 2
```

Adjust the camera in the live preview, then press Enter to freeze the image
and start calibration. Click the four corners in the order shown. Start at the near-left corner, then
near right, far right, and far left. Press Enter to confirm. Press R to redo your
clicks. Recalibrate if the camera moves.

The synthetic LiDAR window shows a top-down view. Green points show range
returns, white circles show the estimated centers of scan clusters, and the
yellow line marks the counting boundary. Cross the line left to right for +1
and right to left for -1. Press Q or Esc to quit.

Add `--scan-log demo.jsonl` to save one scan per line. Each scan contains 721
ranges in metres from 0 to π radians in 0.25° steps, with null for no return.
Angles start toward the right and sweep through forward to the left. The
simulated sensor sits 0.5 m behind the center of the near edge. The timestamp is
Unix time in seconds. Choose a new filename each run.

The camera supplies ground positions for detected people. The simulator models
each as a 36 cm disc and keeps the closest hit on each ray. It omits walls and
furniture. People hidden from the camera or merged into one scan cluster can be
missed. The count uses positions recovered from the range returns.
