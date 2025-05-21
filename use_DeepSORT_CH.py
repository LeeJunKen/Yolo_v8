#!/usr/bin/env python
"""
Deep SORT face tracking (chỉ Color Histogram) + logging + xuất CSV + video
Hiển thị cả Instant FPS và Average FPS trên frame
"""
import sys, os, time, csv
import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf
np.int = int
from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import motmetrics as mm
import pandas as pd

# --- CẤU HÌNH ---
VIDEO_PATH = r"Tracking\video.mp4"
GT_PATH = r"Tracking\gt.txt"
YOLO_WEIGHTS = r"runs/detect/train_detection/weights/best.pt"
MIN_CONFIDENCE = 0.3
MAX_COSINE_DIST = 0.5
NN_BUDGET = 50
MAX_AGE = 9999
N_INIT = 3
DETECT_INTERVAL   = 3
OUTPUT_VIDEO = "out_hist.mp4"
OUTPUT_CSV = "log_hist.csv"


# -----------------

def verify_files():
    for p, d in [(VIDEO_PATH, "Video"), (YOLO_WEIGHTS, "YOLO")]:
        if not os.path.exists(p):
            print(f"{d} not found: {p}");
            sys.exit(1)


def load_mot_gt(gt_path):
    df = pd.read_csv(gt_path, header=None)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    gt = {}
    for _, r in df.iterrows():
        f = int(r.frame)
        x1, y1, w_, h_ = r.x, r.y, r.w, r.h
        box = [x1, y1, x1 + w_, y1 + h_]
        gt.setdefault(f, []).append((int(r.id), box))
    return gt


def color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


gt_dict = load_mot_gt(GT_PATH)
acc = mm.MOTAccumulator(auto_id=True)


def main():
    verify_files()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video");
        sys.exit(1)

    w, h = int(cap.get(3)), int(cap.get(4))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (w, h))
    csv_f = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    writer.writerow(["frame", "timestamp", "inst_fps", "avg_fps", "num_tracks", "track_ids"])

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric, max_age=MAX_AGE, n_init=N_INIT)
    yolo = YOLO(YOLO_WEIGHTS)

    prev_t = time.time()
    start_t = prev_t
    frame_idx = 0

    dets_cache = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # --- Instant FPS ---
        now = time.time()
        inst_fps = 1.0 / (now - prev_t) if now != prev_t else fps_in
        prev_t = now
        # --- Average FPS từ frame 1 đến frame_idx ---
        total_time = now - start_t
        avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        if frame_idx % DETECT_INTERVAL == 1:
            preds = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
            results = preds[0] if preds else None
            bboxes, scores = [], []
            if results and hasattr(results, 'boxes'):
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    if c < MIN_CONFIDENCE: continue
                    bboxes.append([int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)])
                    scores.append(float(c))

            dets = []
            for (x, y, w_, h_), conf in zip(bboxes, scores):
                crop = frame[y:y + h_, x:x + w_]
                if crop.size == 0: continue
                hist = color_histogram(crop)
                dets.append(Detection([x, y, w_, h_], conf, hist))

            dets_cache = np.array(dets)

        tracker.predict()
        tracker.update(dets_cache)

        # --- Vẽ bounding box và thu thập thông tin track ---
        active, pred_boxes, pred_ids = [], [], []
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update != 0: continue
            x1, y1, x2, y2 = tr.to_tlbr().astype(int)
            pred_boxes.append([x1, y1, x2, y2])
            pred_ids.append(tr.track_id)
            active.append(tr.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID_{tr.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Cập nhật MOT metrics ---
        gt_entries = gt_dict.get(frame_idx, [])
        gt_ids = [e[0] for e in gt_entries]
        gt_boxes = [e[1] for e in gt_entries]
        if gt_boxes and pred_boxes:
            iou = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
        acc.update(gt_ids, pred_ids, iou)

        # --- Ghi log và overlay FPS lên frame ---
        ts = time.strftime("%H:%M:%S", time.localtime())
        writer.writerow([frame_idx, ts, f"{inst_fps:.2f}", f"{avg_fps:.2f}",
                         len(active), ";".join(map(str, active))])

        cv2.putText(frame, f"Inst FPS: {inst_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_vid.write(frame)
        cv2.imshow("Hist Only", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Kết thúc ---
    cap.release()
    out_vid.release()
    csv_f.close()
    cv2.destroyAllWindows()

    # --- In báo cáo MOT metrics ---
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames', 'mota', 'idf1','precision','recall','motp',  'num_switches', 'num_fragmentations'],
        name='hist_matching'
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))
    print(f"=== Video processed: {frame_idx} frames")
    print(f"=== Average FPS (total): {avg_fps:.2f}")


if __name__ == "__main__":
    main()
