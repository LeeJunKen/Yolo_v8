#!/usr/bin/env python
"""
Face tracking realtime với YOLOv8 + SORT, kèm:
 - Logging FPS
 - Ghi số lượng tracks active
 - Lưu track IDs theo frame ra CSV
 - Xuất video kết quả
"""
import sys, os, time, csv
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# --- Cấu hình ---
VIDEO_PATH        = r"E:\DoAn\Data\video_2.mp4"
YOLO_FACE_WEIGHTS = r"runs/detect/train3/weights/best.pt"
MIN_CONFIDENCE    = 0.3

# Tham số SORT
IOU_THRESHOLD = 0.3
MIN_HITS      = 3
MAX_AGE       = 30

# Đường dẫn output
OUTPUT_VIDEO    = "output_sort_result.mp4"
OUTPUT_CSV      = "track_ids_log.csv"
# -------------------

def verify_files():
    missing = []
    for path, desc in [
        (VIDEO_PATH, 'Video file'),
        (YOLO_FACE_WEIGHTS, 'YOLO weights')
    ]:
        if not os.path.exists(path):
            missing.append(f"{desc} not found: {path}")
    if missing:
        for m in missing: print(f"Error: {m}")
        sys.exit(1)

def main():
    verify_files()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    # Chuẩn bị VideoWriter (match frame size và FPS input)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

    # Mở CSV để ghi log: frame_idx, timestamp, fps, num_tracks, list(track_ids)
    csv_file = open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "fps", "num_tracks", "track_ids"])

    # Khởi tạo tracker và YOLO
    mot_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    yolo = YOLO(YOLO_FACE_WEIGHTS)

    prev_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Tính FPS đơn giản
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else fps_in
        prev_time = now

        # Face detection
        preds   = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
        results = preds[0] if preds else None

        dets = []
        if results and hasattr(results, 'boxes'):
            xyxy  = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                if conf < MIN_CONFIDENCE: continue
                dets.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

        # Cập nhật SORT
        trackers = mot_tracker.update(np.array(dets)) if dets else mot_tracker.update()

        # Vẽ và ghi log
        active_ids = []
        for x1, y1, x2, y2, track_id in trackers:
            x1,y1,x2,y2,tid = map(int, (x1,y1,x2,y2,track_id))
            active_ids.append(tid)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Ghi CSV
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        csv_writer.writerow([frame_idx, timestamp, f"{fps:.2f}",
                             len(active_ids), ";".join(map(str, active_ids))])

        # Xuất frame ra video
        out_vid.write(frame)

        cv2.imshow("SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_vid.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Finished. Video saved to {OUTPUT_VIDEO}, log saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
