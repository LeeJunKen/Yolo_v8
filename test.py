import cv2
from ultralytics import YOLO
import numpy as np
import time
from deep_sort.deep.feature_extractor import Extractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

model = YOLO('runs/detect/train5/weights/best.pt')

extractor = Extractor('deep_sort/deep/checkpoint/ckpt.t7')

metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
tracker = Tracker(metric, max_age=30)



def color_histogram(image, bins=(8, 8, 8)):
    # Chuyển sang hệ màu HSV (để ổn định với ánh sáng)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

cap = cv2.VideoCapture(0)
processing_width = 320
fps_start_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale_ratio = processing_width / width
    frame_resized = cv2.resize(frame, (processing_width, int(height * scale_ratio)))

    results = model.predict(frame_resized, conf=0.5, verbose=False)[0]

    detections = []

    for bbox in results.boxes:
        x, y, w, h = map(int, bbox.xywh[0])
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        face_img = frame_resized[y1:y1 + h, x1:x1 + w]

        if face_img.size == 0:
            continue

        face_img_resized = cv2.resize(face_img, (64, 128))

        # Extract embedding (CNN embedding)
        # embedding_cnn = extractor([face_img_resized])[0]

        # Color Histogram
        embedding_hist = color_histogram(face_img_resized)

        # combined_embedding = np.hstack((embedding_cnn, embedding_hist))

        detections.append(Detection([x1, y1, w, h], bbox.conf.item(), embedding_hist))

    tracker.predict()
    tracker.update(detections)

    scale_ratio = width / processing_width

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr() * scale_ratio)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = time.time()

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 + Deep SORT Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()