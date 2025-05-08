#!/usr/bin/env python
"""
Deep SORT face tracking using YOLOv8 Face model, kèm:
 - Logging FPS
 - Ghi số lượng tracks active
 - Lưu track IDs theo frame ra CSV
 - Xuất video kết quả
"""
import sys, os, time, csv, urllib.request
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf

from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

np.int = int

# --- Cấu hình ---
VIDEO_PATH          = r"test_class.mp4"
YOLO_FACE_WEIGHTS   = r"runs/detect/train3/weights/best.pt"
DESCRIPTOR_MODEL    = r"deep_sort/mars-small128.pb"
DESCRIPTOR_URL      = 'https://github.com/nwojke/deep_sort/raw/master/resources/networks/mars-small128.pb'
MIN_CONFIDENCE      = 0.3
MAX_COSINE_DISTANCE = 0.5
NN_BUDGET           = 200
MAX_AGE             = 30
N_INIT              = 3

OUTPUT_VIDEO = "output_deepsort_result.mp4"
OUTPUT_CSV   = "deepsort_track_log.csv"
# -------------------

def ensure_descriptor():
    if not os.path.exists(DESCRIPTOR_MODEL):
        os.makedirs(os.path.dirname(DESCRIPTOR_MODEL), exist_ok=True)
        urllib.request.urlretrieve(DESCRIPTOR_URL, DESCRIPTOR_MODEL)

def verify_files():
    for path, desc in [
        (VIDEO_PATH, 'Video file'),
        (YOLO_FACE_WEIGHTS, 'YOLO weights')
    ]:
        if not os.path.exists(path):
            print(f"Error: {desc} not found: {path}")
            sys.exit(1)

def main():
    ensure_descriptor()
    verify_files()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (w, h))

    csv_f = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "timestamp", "fps", "num_tracks", "track_ids"])

    # Khởi tạo Deep SORT
    encoder = gdet.create_box_encoder(DESCRIPTOR_MODEL, batch_size=32)
    metric  = nn_matching.NearestNeighborDistanceMetric(
                  "cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_age=MAX_AGE, n_init=N_INIT)
    yolo    = YOLO(YOLO_FACE_WEIGHTS)

    prev_t = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Tính FPS
        now = time.time()
        fps = 1.0 / (now - prev_t) if now != prev_t else fps_in
        prev_t = now

        # Face detection
        preds = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
        results = preds[0] if preds else None

        bboxes, scores = [], []
        if results and hasattr(results, 'boxes'):
            xyxy  = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                if conf < MIN_CONFIDENCE: continue
                bboxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                scores.append(float(conf))

        # Lấy features và tạo detections
        features = encoder(frame, bboxes) if bboxes else []
        dets = [Detection(b, s, f) for b, s, f in zip(bboxes, scores, features)]

        # Update tracker
        tracker.predict()
        tracker.update(dets)

        # Vẽ và ghi log
        active_ids = []
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update != 0:
                continue
            x1, y1, x2, y2 = tr.to_tlbr().astype(int)
            tid = tr.track_id
            active_ids.append(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        ts = time.strftime("%H:%M:%S", time.localtime())
        writer.writerow([frame_idx, ts, f"{fps:.2f}",
                         len(active_ids), ";".join(map(str, active_ids))])
        out_vid.write(frame)

        cv2.imshow("Deep SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out_vid.release()
    csv_f.close()
    cv2.destroyAllWindows()
    print(f"Saved video: {OUTPUT_VIDEO}, log: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
