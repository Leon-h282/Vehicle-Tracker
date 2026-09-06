

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time


model = YOLO("yolov8n.pt")


def getSrc_points(frame):
    h, w, _ = frame.shape
    
    return np.float32([
        [  450,   300],  # Trên-trái
        [w-450,   300],  # Trên-phải
        [    w, h-200],  # Dưới-phải
        [    0, h-200]   # Dưới-trái
    ])

def draw_track_area(src_points):
    cv2.polylines(
        frame,
        [src_points.astype(np.int32).reshape((-1, 1, 2))],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3)

    for point in src_points:
        point = tuple(map(int, point))

        cv2.circle(
            frame,
            point,
            10, (255, 0, 255), -1
        )


# Chỉ lọc các lớp phương tiện: 2: car, 3: motorcycle, 5: bus, 7: truck (theo COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7]

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps==0:
    fps = 30

prev_time  = time.time()
frame_time = 1.0/fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame hoặc video đã kết thúc.")
        break

    src_points = getSrc_points(frame)
    draw_track_area(src_points)

    result = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    cv2.imshow("camera", frame)

    dt = time.time() - prev_time
    wait = max(1, int((frame_time - dt) * 1e3))
    key = cv2.waitKey(wait) & 0xFF
    if key == ord('q'):
        break
    prev_time = time.time()
