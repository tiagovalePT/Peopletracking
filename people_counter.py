"""Count people crossing the middle of a webcam frame."""
import argparse
import os
import sys
import time
from pathlib import Path


class Counter:
    def __init__(self):
        self.net = 0
        self.last_side = {}

    def update(self, track_id, offset, margin):
        # Keep the last confirmed side while the center is near the line.
        if abs(offset) <= margin:
            return
        side = 1 if offset > 0 else -1
        previous = self.last_side.get(track_id, side)
        if previous != side:
            self.net += side
        self.last_side[track_id] = side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    import cv2

    project = Path(__file__).resolve().parent
    os.environ.setdefault("YOLO_CONFIG_DIR", str(project / ".yolo"))
    from ultralytics import YOLO

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture()
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
            raise SystemExit(
                f"Could not read camera {args.camera}. Check camera permissions "
                "or try --camera 1."
            )

        model = YOLO(str(project / "yolo11n.pt"))
        counter = Counter()
        while ok:
            height, width = frame.shape[:2]
            line = width // 2
            margin = max(4, round(width * 0.02))
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is not None and boxes.is_track:
                for track_id, box in zip(
                    boxes.id.int().tolist(), boxes.xyxy.int().tolist()
                ):
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    counter.update(track_id, center[0] - line, margin)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
                    cv2.circle(frame, center, 4, (80, 220, 80), -1)

            cv2.line(frame, (line, 0), (line, height - 1), (0, 220, 255), 2)
            label = f"Net count: {counter.net:+d}" if counter.net else "Net count: 0"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.8, width / 1000)
            thickness = max(2, round(scale * 2))
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )
            padding = max(10, width // 100)
            cv2.rectangle(
                frame, (padding, padding),
                (text_width + 3 * padding, text_height + baseline + 3 * padding),
                (25, 25, 25), -1,
            )
            cv2.putText(
                frame, label, (2 * padding, text_height + 2 * padding),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
            cv2.imshow("People Counter", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            ok, frame = camera.read()
        print(f"Final net count: {counter.net}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
