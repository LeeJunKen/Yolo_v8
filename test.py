import cv2
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Khởi tạo YOLO và DeepSort
model = YOLO("runs/detect/train5/weights/best.pt")
tracker = DeepSort(max_age=30)
conf_threshold = 0.5

# Queue cho frame
frame_queue = queue.Queue(maxsize=5)   # Giới hạn số frame trong queue để tránh tràn bộ nhớ
result_queue = queue.Queue(maxsize=5)

def frame_reader(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)  # Dấu hiệu kết thúc

def detection_and_tracking():
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break

        results = model.predict(source=frame, conf=0.3, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                bbox = [x1, y1, x2 - x1, y2 - y1]
                boxes.append([bbox, conf, "0"])
        # tracks = tracker.update_tracks(boxes, frame=frame)
        # Gắn kết quả tracking (các track đã update) vào frame để hiển thị
        # for track in tracks:
        #     if track.is_confirmed():
        #         track_id = track.track_id
        #         ltrb = track.to_ltrb()
        #         x1, y1, x2, y2 = map(int, ltrb)
        #         label = f"ID: {track_id}"
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(frame, label, (x1, y1 - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        result_queue.put(frame)

def display_results():
    prev_time = time.time()
    while True:
        frame = result_queue.get()
        if frame is None:
            break
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Khởi tạo các thread
reader_thread = threading.Thread(target=frame_reader, args=("video/test.mp4",))
detector_thread = threading.Thread(target=detection_and_tracking)
display_thread = threading.Thread(target=display_results)

reader_thread.start()
detector_thread.start()
display_thread.start()

reader_thread.join()
detector_thread.join()
display_thread.join()
