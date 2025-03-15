import cv2
from ultralytics import YOLO
import numpy as np
from deep_sort.deep.feature_extractor import Extractor
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

# Load YOLOv8 face detection model
model = YOLO('runs/detect/train5/weights/best.pt')

# Initialize Deep SORT embedding extractor
extractor = Extractor('deep_sort/deep/checkpoint/ckpt.t7')

# Initialize Deep SORT tracker
metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
tracker = Tracker(metric)

# Video input (webcam hoặc video file)
cap = cv2.VideoCapture("video/test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    results = model.predict(frame, conf=0.5)[0]

    detections = []

    for bbox in results.boxes:
        x, y, w, h = map(int, bbox.xywh[0])
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        face_img = frame[y1:y1 + h, x1:x1 + w]

        if face_img.size == 0:
            continue

        # Extract embedding
        embedding = extractor([face_img])[0]

        # Create Detection
        detection = Detection([x1, y1, w, h], bbox.conf.item(), embedding)
        detections.append(detection)

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    # Display tracking results
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 + Deep SORT Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()