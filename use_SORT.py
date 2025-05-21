#!/usr/bin/env python
"""
Face tracking realtime với YOLOv8 + SORT, kèm:
 - Logging FPS
 - Ghi số lượng tracks active
 - Lưu track IDs theo frame ra CSV
 - Xuất video kết quả
 - Tính MOT metrics
"""
import sys, os, time, csv

import cv2
import numpy as np
import pandas as pd
import motmetrics as mm

from ultralytics import YOLO
from sort import Sort

# --- Cấu hình ---
VIDEO_PATH        = r"Tracking\video.mp4"
GT_PATH           = r"Tracking\gt.txt"      # ground truth file
YOLO_FACE_WEIGHTS = r"runs/detect/train_detection/weights/best.pt"
MIN_CONFIDENCE    = 0.3

# Tham số SORT
IOU_THRESHOLD = 0.3
MIN_HITS      = 3
MAX_AGE       = 30
DETECT_INTERVAL   = 3
# Đường dẫn output
OUTPUT_VIDEO = "output_sort_result.mp4"
OUTPUT_CSV   = "track_ids_log.csv"
# -------------------

def verify_files():
    missing = []
    for path, desc in [
        (VIDEO_PATH, 'Video file'),
        (YOLO_FACE_WEIGHTS, 'YOLO weights'),
        (GT_PATH, 'Ground Truth file')
    ]:
        if not os.path.exists(path):
            missing.append(f"{desc} not found: {path}")
    if missing:
        for m in missing:
            print("Error:", m)
        sys.exit(1)

def load_mot_gt(gt_path):
    """
    Load ground truth MOTChallenge format (frame,id,x,y,w,h,conf,class,vis)
    into a dict: frame_no -> [(id, [x1,y1,x2,y2]), ...]
    """
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

    # --- Load GT và khởi tạo MOTAccumulator ---
    gt_dict = load_mot_gt(GT_PATH)
    acc     = mm.MOTAccumulator(auto_id=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video"); sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_in, (width, height))

    csv_file = open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "fps", "num_tracks", "track_ids"])

    mot_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
    yolo = YOLO(YOLO_FACE_WEIGHTS)

    prev_time = time.time()
    frame_idx = 0

    dets_cache = np.empty((0, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Tính FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else fps_in
        prev_time = now

        if frame_idx % DETECT_INTERVAL == 1:
            # Face detection
            preds   = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
            results = preds[0] if preds else None

            dets = []
            if results and hasattr(results, 'boxes'):
                xyxy  = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    if conf < MIN_CONFIDENCE:
                        continue
                    dets.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

            dets_cache = np.array(dets)

        # Cập nhật SORT
        trackers = mot_tracker.update(dets_cache) if len(dets) else mot_tracker.update()

        # Vẽ và ghi log
        pred_boxes, pred_ids = [], []
        for t in trackers:
            x1, y1, x2, y2, track_id = t
            x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, track_id))
            pred_boxes.append([x1, y1, x2, y2])
            pred_ids.append(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID{tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Ghi CSV
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        csv_writer.writerow([
            frame_idx, timestamp, f"{fps:.2f}",
            len(pred_ids), ";".join(map(str, pred_ids))
        ])

        # Cập nhật MOT-metrics
        gt_entries = gt_dict.get(frame_idx, [])
        gt_ids   = [e[0] for e in gt_entries]
        gt_boxes = [e[1] for e in gt_entries]
        if gt_boxes and pred_boxes:
            iou_mat = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            iou_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
        acc.update(gt_ids, pred_ids, iou_mat)

        # Xuất video
        out_vid.write(frame)
        cv2.imshow("SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_vid.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # Tính và in MOT metrics summary
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=['num_frames', 'mota', 'motp', 'idf1', 'num_switches', 'num_fragmentations'],
        name='sort_tracking'
    )
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Log CSV:       {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
