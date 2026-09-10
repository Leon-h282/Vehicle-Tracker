

import cv2
import numpy as np
from ultralytics import YOLO
import time


# Màu (BGR)
COL_BLUE   = (255, 0, 0)
COL_GREEN  = (0, 255, 0)
COL_WHITE  = (255, 255, 255)
COL_YELLOW = (0, 180, 180)


# Bounding box font size
bbox_font_size = 0.5


# Load model YOLOv8
model = YOLO("yolov8n.pt")


# Chỉ lọc các lớp phương tiện (theo COCO dataset):
#   2: car
#   3: motorcycle
#   5: bus
#   7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Đường dẫn video
video_path = "video.mp4"

# Đối tượng đọc khung hình từ video
cap = cv2.VideoCapture(video_path)


# -----------------------------------------------
# TÍNH TOÁN TỐC ĐỘ CAMERA
# Số frame đã chiếu trong 1 giây = FPS * 1.0s
# -----------------------------------------------
# Lấy FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps==0:
    fps = 30

# Thời gian của 1 frame
frame_time = 1.0/fps


# -----------------------------------------
# KHOANG VÙNG THEO DÕI TỐC ĐỘ XE
# 
# - Khoanh vùng trên màn hình
# - Xác định kích thước làn đường thực tế
# - Chuyển đổi đơn vị px -> meter
# -----------------------------------------

# Khoanh vùng trên màn hình
def get_src_points(frame: np.ndarray) -> np.float32:
    h, w = frame.shape[:2]
    
    return np.float32([
        [  450,   300],  # Trên-trái
        [w-450,   300],  # Trên-phải
        [    w, h-200],  # Dưới-phải
        [    0, h-200]   # Dưới-trái
    ])


# Kích thước làn đường trên thực tế (đv: mét)
REAL_WIDTH  = 32    # Chiều rộng làn đường
REAL_HEIGHT = 80    # Chiều dài đoạn đường được khoanh vùng

# Phóng các điểm lên màn hình
dst_point = np.float32([
    [0, 0],                     # Trên-trái
    [REAL_WIDTH, 0],            # Trên-phải
    [REAL_WIDTH, REAL_HEIGHT],  # Dưới-phải
    [0, REAL_HEIGHT]            # Dưới-trái
])

# Lưu tọa độ thực tế ở frame trước của từng xe
track_history = {}

# Chuyển pixel sang mét
def pixel_to_meters(point: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, M)
    return transformed


# Vẽ đường khoanh vùng
def draw_track_area(frame: np.ndarray, src_points: np.ndarray) -> None:
    overlay = frame.copy()
    alpha = 0.2

    # Tô màu vùng theo dõi
    cv2.fillPoly(
        overlay, [src_points.astype(np.int32)], COL_GREEN
    )

    # Chồng lớp overlay để tạo hiệu ứng transparent
    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0, frame
    )


def main():
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame hoặc video đã kết thúc.")
            break

        src_points = get_src_points(frame)

        # Ma trận chuyển đổi perspective
        M = cv2.getPerspectiveTransform(src_points, dst_point)

        result = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            verbose=False
        )

        draw_track_area(frame, src_points)

        if result[0].boxes and result[0].boxes.id is not None:
            # Tọa độ 4 góc bounding box
            boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

            # ID của xe đang đc theo dõi
            track_ids = result[0].boxes.id.cpu().numpy().astype(int)

            # Class ID
            classes = result[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box

                # Lấy trung điểm đáy của bounding box
                bottom_center = ((x1 + x2)/2, y2)

                if (
                    bottom_center[1] >= src_points[0][1] and
                    bottom_center[1] <= src_points[2][1]
                ):

                    # Chuyển pixel sang mét
                    real_pos = pixel_to_meters(bottom_center, M)

                    speed_kmh = 0
                    if track_id in track_history:
                        prev_pos = track_history[track_id]

                        # Quãng đường đã di chuyển
                        dist = np.linalg.norm(real_pos - prev_pos)

                        # Tính tốc độ
                        speed_mps = dist / frame_time
                        speed_kmh = speed_mps * 3.6    # Đổi đơn vị sang km/h

                    track_history[track_id] = real_pos

                    # -------------------------------------
                    # VẼ BBOX VÀ HIỂN THỊ TỐC ĐỘ
                    # -------------------------------------
                    # Tên lớp (loại phương tiện)
                    class_name = model.names[cls_id]

                    bbox_col = COL_BLUE if speed_kmh <= 50 else COL_YELLOW

                    # Bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        bbox_col,
                        thickness=1
                    )

                    # Ghi nhãn
                    label = f"{class_name} ID:{track_id} | {int(speed_kmh)} km/h"

                    # Lấy kích thước chữ
                    (text_w, text_h), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        bbox_font_size, thickness=1
                    )

                    # Nền cho nhãn
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_h - 5),
                        (x1 + text_w + 5, y1),
                        bbox_col,
                        thickness=-1
                    )

                    # Hiển thị nhãn
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        bbox_font_size, COL_WHITE, 1
                    )

        cv2.imshow("camera", frame)

        # Hiển thị khung hình theo tốc độ thực
        dt = time.time() - prev_time
        wait = max(1, int((frame_time - dt) * 1e3))

        key = cv2.waitKey(wait) & 0xFF
        if key == ord('q'):
            break
        prev_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
