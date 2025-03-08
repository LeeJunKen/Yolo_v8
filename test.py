import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

############################
# 1. Khởi tạo YOLOv8 & Deep SORT
############################

# YOLOv8 - Model phát hiện (ví dụ: YOLOv8n, YOLOv8s,...)
model = YOLO("runs/detect/train5/weights/best.pt")  # thay bằng đường dẫn hoặc model tùy ý

tracker = DeepSort(max_age=30)
conf_threshold = 0.5
tracking_class = 2
cap = cv2.VideoCapture("video/test.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # #Detect
    results = model.predict(source=frame, conf=0.3, verbose=False)
    #
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # print(x1,y1,x2,y2)
            conf = float(box.conf[0])

            if tracking_class is None:
                if conf < conf_threshold:
                    continue
            else:
                if conf < conf_threshold:
                    continue

            boxes.append([[x1, y1, x2-x1, y2-y1], conf, "0"])
    tracks = tracker.update_tracks(boxes, frame=frame)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            # color = colors[class_id]

            label = "0"
            B, G, R = 0, 255, 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label,(x1 + 5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()