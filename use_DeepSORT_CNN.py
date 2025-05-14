#!/usr/bin/env python
"""
Deep SORT face tracking (embedding CNN từ MARS) + logging + xuất CSV + video + MOT metrics
"""
import sys, os, time, csv
import cv2
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf
np.int = int

from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

import motmetrics as mm

# --- CẤU HÌNH ---
VIDEO_PATH        = r"Tracking\video.mp4"
GT_PATH           = r"Tracking\gt.txt"           # đường dẫn GT MOTChallenge
YOLO_WEIGHTS      = r"runs/detect/train_detection/weights/best.pt"
DESCRIPTOR_MODEL  = r"deep_sort/mars-small128.pb"
MIN_CONFIDENCE    = 0.3
MAX_COSINE_DIST   = 0.5
NN_BUDGET         = 200
MAX_AGE           = 30
N_INIT            = 3
OUTPUT_VIDEO      = "out_cnn.mp4"
OUTPUT_CSV        = "log_cnn.csv"
# -----------------

def verify_files():
    for p, desc in [
        (VIDEO_PATH,       "Video"),
        (GT_PATH,          "Ground Truth"),
        (YOLO_WEIGHTS,     "YOLO weights"),
        (DESCRIPTOR_MODEL, "MARS model")
    ]:
        if not os.path.exists(p):
            print(f"Error: {desc} not found: {p}")
            sys.exit(1)

def load_mot_gt(gt_path):
    """Load GT MOTChallenge format into dict[frame] -> list of (id, [x1,y1,x2,y2])"""
    df = pd.read_csv(gt_path, header=None)
    df.columns = ["frame","id","x","y","w","h","conf","class","vis"]
    gt = {}
    for _, r in df.iterrows():
        f = int(r.frame)
        x1, y1, w, h = r.x, r.y, r.w, r.h
        box = [x1, y1, x1 + w, y1 + h]
        gt.setdefault(f, []).append((int(r.id), box))
    return gt

def main():
    verify_files()

    # --- Load GT và khởi tạo MOT accumulator ---
    gt_dict = load_mot_gt(GT_PATH)
    acc = mm.MOTAccumulator(auto_id=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video"); sys.exit(1)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (w, h))
    csv_f = open(OUTPUT_CSV,'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    writer.writerow(["frame","timestamp","fps","num_tracks","track_ids"])

    # Khởi tạo tracker và encoder
    encoder = gdet.create_box_encoder(DESCRIPTOR_MODEL, batch_size=32)
    metric  = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric, max_age=MAX_AGE, n_init=N_INIT)
    yolo    = YOLO(YOLO_WEIGHTS)

    prev_t, frame_idx = time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Tính FPS
        now = time.time()
        fps = 1.0/(now - prev_t) if now != prev_t else fps_in
        prev_t = now

        # 1) Detect
        preds   = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
        results = preds[0] if preds else None
        bboxes, scores = [], []
        if results and hasattr(results, 'boxes'):
            xyxy  = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), conf in zip(xyxy, confs):
                if conf < MIN_CONFIDENCE:
                    continue
                bboxes.append([int(x1),int(y1),int(x2)-int(x1),int(y2)-int(y1)])
                scores.append(float(conf))

        # 2) Extract CNN features
        features = encoder(frame, bboxes) if bboxes else []

        # 3) Build detections
        dets = []
        for b, s, f in zip(bboxes, scores, features):
            dets.append(Detection(b, s, f))

        # 4) Track
        tracker.predict()
        tracker.update(dets)

        # 5) Vẽ và thu thập pred
        active_ids, pred_boxes = [], []
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update != 0:
                continue
            x1,y1,x2,y2 = tr.to_tlbr().astype(int)
            tid = tr.track_id
            active_ids.append(tid)
            pred_boxes.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID_{tid}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 6) Ghi CSV
        ts = time.strftime("%H:%M:%S", time.localtime())
        writer.writerow([frame_idx, ts, f"{fps:.2f}",
                         len(active_ids), ";".join(map(str, active_ids))])

        # 7) Cập nhật MOT accumulator
        gt_entries = gt_dict.get(frame_idx, [])
        gt_ids   = [e[0] for e in gt_entries]
        gt_boxes = [e[1] for e in gt_entries]
        if gt_boxes and pred_boxes:
            iou = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            iou = np.zeros((len(gt_ids), len(pred_boxes)), dtype=float)
        acc.update(gt_ids, active_ids, iou)

        # 8) Ghi video và hiển thị
        out_vid.write(frame)
        cv2.imshow("CNN Only Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_vid.release()
    csv_f.close()
    cv2.destroyAllWindows()

    # --- Tính MOT metrics & in summary ---
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames','mota','motp','idf1',
            'num_switches','num_fragmentations'
        ],
        name='cnn_only'
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))

    print(f"Saved {OUTPUT_VIDEO}, {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
