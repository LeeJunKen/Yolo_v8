import cv2
import numpy as np
import time
from ultralytics import YOLO

# Import các module từ Deep SORT
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from tools.generate_detections import create_box_encoder

# ---------------------------
# 1. Khởi tạo detector và Deep SORT
# ---------------------------

# Load model YOLO để phát hiện khuôn mặt
model = YOLO("runs/detect/train5/weights/best.pt")
cap = cv2.VideoCapture("video/test.mp4")

# Deep SORT: thiết lập metric và tracker
max_cosine_distance = 0.4
nn_budget = None
# Đường dẫn đến tệp mô hình trích xuất đặc trưng (appearance descriptor)
model_filename = 'mars-small128.pb'
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
# Hàm encoder sẽ trích xuất feature cho các bounding box
encoder = create_box_encoder(model_filename, batch_size=1)

# ---------------------------
# 2. Vòng lặp xử lý video (tracking-by-detection)
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bounding box sử dụng YOLO (định dạng [x1, y1, x2, y2])
    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = []
    confidences = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

    # Nếu có detection, chuyển đổi định dạng bounding box sang [x, y, w, h]
    detections = []
    if len(boxes) > 0:
        # Chuyển đổi: [x, y, x2, y2] -> [x, y, w, h]
        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            bboxes.append([x1, y1, w, h])
        bboxes = np.array(bboxes)

        # Tính các feature appearance cho từng detection
        features = encoder(frame, bboxes)
        # Tạo danh sách các đối tượng Detection của Deep SORT
        detections = [Detection(bbox, conf, feature)
                      for bbox, conf, feature in zip(bboxes, confidences, features)]

    # Cập nhật tracker với các detection mới
    tracker.predict()
    tracker.update(detections)

    # Vẽ bounding box và ID cho các tracker đã được xác nhận
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()  # Chuyển sang định dạng (x1, y1, x2, y2)
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame kết quả
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
