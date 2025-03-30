import cv2
from ultralytics import YOLO
import numpy as np
import time
import motmetrics as mm
from deep_sort.deep.feature_extractor import Extractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

model = YOLO('runs/detect/train5/weights/best.pt')
extractor = Extractor('deep_sort/deep/checkpoint/ckpt.t7')

metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.5)
tracker = Tracker(metric, max_age=30)

def color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

cap = cv2.VideoCapture("video/test.mp4")
processing_width = 320

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracking.mp4', fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

fps_start_time = time.time()
frame_count = 0
fps = 0

# Setup motmetrics
acc = mm.MOTAccumulator(auto_id=True)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale_ratio = processing_width / width
    frame_resized = cv2.resize(frame, (processing_width, int(height * scale_ratio)))

    results = model.predict(frame_resized, conf=0.5, verbose=False)[0]

    detections = []
    pred_boxes, pred_ids = [], []

    for bbox in results.boxes:
        x, y, w, h = map(int, bbox.xywh[0])
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        face_img = frame_resized[y1:y1 + h, x1:x1 + w]

        if face_img.size == 0:
            continue

        face_img_resized = cv2.resize(face_img, (64, 128))
        embedding_cnn = extractor([face_img_resized])[0]
        embedding_hist = color_histogram(face_img_resized)
        combined_embedding = np.hstack((embedding_cnn, embedding_hist))

        detections.append(Detection([x1, y1, w, h], bbox.conf.item(), combined_embedding))

    tracker.predict()
    tracker.update(detections)

    scale_ratio = width / processing_width

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr() * scale_ratio)
        pred_boxes.append([x1, y1, x2, y2])
        pred_ids.append(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính FPS
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = time.time()

    # cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Ghi video kết quả
    out.write(frame)

    # (Cần dữ liệu GT để tính chính xác)
    # gt_ids = []
    # gt_boxes = []
    # distance_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    # acc.update(gt_ids, pred_ids, distance_matrix)

    frame_idx += 1

    # cv2.imshow('YOLOv8 + Deep SORT Face Tracking', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

# Hiển thị kết quả tracking metrics
# mh = mm.metrics.create()
# summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches'], name='YOLO_DeepSORT')
# print(summary)
