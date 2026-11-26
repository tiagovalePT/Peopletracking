"""Generate a synthetic 2D LiDAR scan from a calibrated webcam."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from people_counter import Counter


ANGLES = np.linspace(0, np.pi, 721)
RAYS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
PERSON_RADIUS = 0.18


def simulate_scan(people, origin, max_range):
    """Return the nearest ray intersection in metres, or infinity for no return."""
    ranges = np.full(len(ANGLES), np.inf)
    for person in people:
        offset = person - origin
        along = RAYS @ offset
        discriminant = PERSON_RADIUS**2 - (offset @ offset - along**2)
        near = along - np.sqrt(np.maximum(discriminant, 0))
        hits = (discriminant >= 0) & (near > 0) & (near <= max_range)
        ranges[hits] = np.minimum(ranges[hits], near[hits])
    return ranges


def scan_centers(ranges, origin):
    """Estimate person centers from contiguous clusters of scan returns."""
    indices = np.flatnonzero(np.isfinite(ranges))
    if not len(indices):
        return []
    points = RAYS[indices] * ranges[indices, None]
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cuts = np.flatnonzero((np.diff(indices) > 1) | (gaps > 0.16)) + 1
    centers = []
    for cluster in np.split(points, cuts):
        if len(cluster) < 3:
            continue
        surface = cluster.mean(axis=0)
        centers.append(origin + surface * (1 + 0.14 / np.linalg.norm(surface)))
    return centers


class ScanCounter:
    def __init__(self, line_x):
        self.line_x = line_x
        self.counter = Counter()
        self.tracks = {}
        self.frame = 0
        self.next_id = 0

    def update(self, centers):
        self.frame += 1
        for track_id, (_, last_seen) in list(self.tracks.items()):
            if self.frame - last_seen > 8:
                del self.tracks[track_id]
                self.counter.last_side.pop(track_id, None)

        candidates = sorted(
            (np.linalg.norm(center - previous), track_id, index)
            for track_id, (previous, _) in self.tracks.items()
            for index, center in enumerate(centers)
        )
        assigned = {}
        used = set()
        for distance, track_id, index in candidates:
            if distance > 0.65:
                break
            if track_id not in used and index not in assigned:
                assigned[index] = track_id
                used.add(track_id)

        for index, center in enumerate(centers):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            self.tracks[track_id] = (center, self.frame)
            self.counter.update(track_id, center[0] - self.line_x, 0.12)
        return self.counter.net


def preview_size(frame):
    height, width = frame.shape[:2]
    scale = min(1, 960 / width, 650 / height)
    return round(width * scale), round(height * scale)


def wait_for_calibration(camera, frame):
    window = "Position camera"
    try:
        while True:
            preview = cv2.resize(frame, preview_size(frame))
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 36), (25, 25, 25), -1)
            cv2.putText(
                preview, "Adjust camera, then press Enter to calibrate",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0:
                raise SystemExit("Calibration cancelled.")
            if key in (10, 13):
                return frame
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Camera feed stopped before calibration.")
    finally:
        cv2.destroyWindow(window)


def calibrate(frame, width, depth):
    window = "Floor calibration"
    preview = cv2.resize(frame, preview_size(frame))
    points = []

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    labels = ("near left", "near right", "far right", "far left")
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            image = preview.copy()
            for point in points:
                cv2.circle(image, point, 6, (0, 220, 255), -1)
            if len(points) == 4:
                cv2.polylines(image, [np.array(points)], True, (0, 220, 255), 2)
                message = "Enter: confirm    R: reset    Esc: quit"
            else:
                message = f"Click {labels[len(points)]} floor corner    R: reset"
            cv2.rectangle(image, (0, 0), (image.shape[1], 36), (25, 25, 25), -1)
            cv2.putText(image, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.imshow(window, image)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0:
                raise SystemExit("Calibration cancelled.")
            if key == ord("r"):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                polygon = np.array(points, dtype=np.float32)
                if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 100:
                    print("Use four clearly spaced corners in the order shown.")
                    points.clear()
                    continue
                polygon *= np.array([frame.shape[1] / preview.shape[1],
                                     frame.shape[0] / preview.shape[0]])
                floor = np.float32([(0, 0), (width, 0), (width, depth), (0, depth)])
                return cv2.getPerspectiveTransform(polygon, floor)
    finally:
        cv2.destroyWindow(window)


def draw_scan(ranges, centers, origin, width, depth, net):
    canvas = np.full((560, 680, 3), (28, 28, 28), dtype=np.uint8)
    scale = min(580 / width, 405 / (depth - origin[1]))

    def pixel(point):
        x, y = point
        return round(340 + (x - width / 2) * scale), round(500 - (y - origin[1]) * scale)

    cv2.rectangle(canvas, pixel((0, depth)), pixel((width, 0)), (110, 110, 110), 1)
    cv2.line(canvas, pixel((width / 2, 0)), pixel((width / 2, depth)), (0, 220, 255), 2)
    cv2.circle(canvas, pixel(origin), 6, (255, 190, 80), -1)
    valid = np.isfinite(ranges)
    for point in origin + RAYS[valid] * ranges[valid, None]:
        cv2.circle(canvas, pixel(point), 2, (130, 245, 140), -1)
    for center in centers:
        cv2.circle(canvas, pixel(center), 8, (255, 255, 255), 1)
    cv2.putText(canvas, f"LiDAR net count: {net}", (22, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(canvas, "Green: scan returns   White: tracked clusters   Q: quit",
                (22, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    return canvas


def metres(value):
    distance = float(value)
    if not math.isfinite(distance) or distance <= 0:
        raise argparse.ArgumentTypeError("Enter a positive distance in metres.")
    return distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=metres, required=True, help="Floor rectangle width in metres")
    parser.add_argument("--depth", type=metres, required=True, help="Floor rectangle depth in metres")
    parser.add_argument("--scan-log", type=Path, help="Write scans to a new JSON Lines file")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".yolo"))
    from ultralytics import YOLO

    camera = cv2.VideoCapture()
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    try:
        deadline = time.monotonic() + 5
        while True:
            if not camera.isOpened():
                camera.open(args.camera, backend)
            ok, frame = camera.read()
            if ok or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        if not ok:
            raise SystemExit(f"Could not read camera {args.camera}.")

        frame = wait_for_calibration(camera, frame)
        mapping = calibrate(frame, args.width, args.depth)
        model = YOLO(str(project / "yolo11n.pt"))
        origin = np.array([args.width / 2, -0.5])
        max_range = math.hypot(args.width / 2, args.depth + 0.5) + PERSON_RADIUS
        counter = ScanCounter(args.width / 2)
        log_context = args.scan_log.open("x", encoding="utf-8") if args.scan_log else nullcontext()
        with log_context as log:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                stamp = time.time()
                boxes = model.predict(frame, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
                people = []
                if len(boxes):
                    feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
                    floor = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), mapping)[:, 0]
                    for (x1, y1, x2, y2), (x, y) in zip(boxes, floor):
                        if y2 >= frame.shape[0] - 2 or not (0 <= x <= args.width and 0 <= y <= args.depth):
                            continue
                        people.append(np.array([x, y]))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (80, 220, 80), 2)

                ranges = simulate_scan(people, origin, max_range)
                centers = scan_centers(ranges, origin)
                net = counter.update(centers)
                if log:
                    record = dict(timestamp=stamp, angle_min=0, angle_max=math.pi,
                                  angle_increment=math.pi / (len(ANGLES) - 1),
                                  range_min=0, range_max=max_range, origin_m=origin.tolist(),
                                  ranges=[float(r) if np.isfinite(r) else None for r in ranges],
                                  net_count=net)
                    log.write(json.dumps(record, allow_nan=False) + "\n")

                cv2.imshow("Camera input", cv2.resize(frame, preview_size(frame)))
                cv2.imshow("Synthetic 2D LiDAR", draw_scan(ranges, centers, origin, args.width, args.depth, net))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        print(f"Final LiDAR net count: {counter.counter.net}")
    except FileExistsError:
        raise SystemExit(f"Scan log already exists: {args.scan_log}. Choose a new filename.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
