import cv2
import numpy as np
from ultralytics import YOLO

# Import các module Deep SORT cần thiết
# Chú ý đường dẫn phù hợp với nơi bạn đặt deep_sort
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# (Tùy chọn) Nếu bạn cần trích xuất appearance feature (cho re-identification),
# hãy sử dụng một encoder. Ví dụ, nếu dùng generate_detections.py, bạn có thể:
# from deep_sort.tools.generate_detections import create_box_encoder
# encoder = create_box_encoder("mars-small128.pb", batch_size=1)

#################################
# 1. Khởi tạo YOLOv8 & Deep SORT
#################################

# Khởi tạo YOLOv8 (ví dụ, mô hình YOLOv8n)
model = YOLO("runs/detect/train5/weights/best.pt")

# Thiết lập tham số cho deep sort:
max_cosine_distance = 0.4
nn_budget = None
metric = NearestNeighborDistanceMetric(metric="cosine",
                                       matching_threshold=max_cosine_distance,
                                       budget=nn_budget)

# Khởi tạo tracker
tracker = Tracker(metric)

# (Nếu có) khởi tạo hàm encoder (trích xuất appearance feature) tại đây
# encoder = ...

#################################
# 2. Hàm chuyển YOLO bbox -> Deep SORT
#################################
def xyxy_to_xywh(x1, y1, x2, y2):
    """Chuyển bounding box YOLO (x1, y1, x2, y2) -> (x, y, w, h)."""
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

#################################
# 3. Vòng lặp đọc video và tracking
#################################
import math


def run_tracking(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện bằng YOLO
        results = model.predict(source=frame, conf=0.3, verbose=False)
        detections_yolo = []
        confidences = []

        frame_h, frame_w = frame.shape[:2]

        # Lấy bounding box và confidence từ YOLO
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Loại bỏ bounding box ngược hoặc bằng nhau
                if x2 <= x1 or y2 <= y1:
                    continue

                # Giới hạn bbox trong kích thước frame
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y2 = max(0, min(y2, frame_h - 1))

                # Nếu sau clamp vẫn không hợp lệ
                if x2 <= x1 or y2 <= y1:
                    continue

                # Kiểm tra NaN/Inf
                if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2, conf]):
                    continue

                detections_yolo.append([x1, y1, x2, y2])
                confidences.append(conf)

        # Tạo Detection cho Deep SORT
        ds_detections = []
        if len(detections_yolo) > 0:
            ds_bboxes = []
            valid_confidences = []
            for bbox, conf in zip(detections_yolo, confidences):
                x1, y1, x2, y2 = bbox
                x, y, w, h = xyxy_to_xywh(x1, y1, x2, y2)

                if w <= 0 or h <= 0:
                    continue

                ds_bboxes.append([x, y, w, h])
                valid_confidences.append(conf)

            if len(ds_bboxes) > 0:
                ds_bboxes = np.array(ds_bboxes, dtype=np.float32)
                # Nếu chưa dùng encoder, tạm gán feature = zeros
                features = np.zeros((len(ds_bboxes), 128), dtype=np.float32)

                ds_detections = [
                    Detection(ds_bbox, conf, feat)
                    for ds_bbox, conf, feat in zip(ds_bboxes, valid_confidences, features)
                ]

        # Cập nhật tracker
        tracker.predict()
        if ds_detections:  # nếu có bbox hợp lệ
            tracker.update(ds_detections)
        else:
            # Gọi update với list trống để tracker cập nhật track cũ (nếu có)
            tracker.update([])

        # Vẽ bounding box và ID
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Chạy thử trên video
    run_tracking(r"E:\DoAn\Data\6313772697550.mp4")
