import cv2
import time
from ultralytics import YOLO
from prlab.projects.FaceTracking.HungarianMatching import HungarianMatching   # hoặc từ deep_sort.HungarianMatching import HungarianMatching

# Khởi tạo YOLOv8 (thay đường dẫn model nếu cần)
model = YOLO("runs/detect/train5/weights/best.pt")
cap = cv2.VideoCapture("video/test.mp4")

# Khởi tạo HungarianMatching
hungarian = HungarianMatching()

# Biến để kiểm soát việc gọi start (frame đầu tiên) hay update (các frame sau)
first_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dùng YOLOv8 để phát hiện khuôn mặt
    results = model.predict(source=frame, conf=0.3, verbose=False)
    bboxes = []
    for r in results:
        for box in r.boxes:
            # Lấy bounding box dạng [x, y, x+w, y+h]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            # Nếu cần có thêm filter theo confidence...
            bboxes.append([x1, y1, w, h])

    # Nếu có detection, gọi HungarianMatching
    if len(bboxes) > 0:
        if first_frame:
            key_ids = hungarian.start(frame, bboxes)
            first_frame = False
        else:
            key_ids = hungarian.update(frame, bboxes)

    # Vẽ các bounding box và nhãn từ HungarianMatching
    hungarian.draw_bboxes(frame)

    # (Tùy chọn) Hiển thị FPS
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
