#!/usr/bin/env python
"""
Deep SORT face tracking (chỉ Color Histogram) + logging + xuất CSV + video
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

# --- CẤU HÌNH ---
VIDEO_PATH     = r"E:\DoAn\Data\video_2.mp4"
YOLO_WEIGHTS   = r"runs/detect/train3/weights/best.pt"
MIN_CONFIDENCE = 0.3
MAX_COSINE_DIST= 0.5
NN_BUDGET      = 200
MAX_AGE        = 200
N_INIT         = 3
OUTPUT_VIDEO   = "out_hist.mp4"
OUTPUT_CSV     = "log_hist.csv"
# -----------------

def verify_files():
    for p, d in [(VIDEO_PATH,"Video"), (YOLO_WEIGHTS,"YOLO")]:
        if not os.path.exists(p):
            print(f"{d} not found: {p}"); sys.exit(1)

def color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = cv2.normalize(hist,hist).flatten()
    return hist

def main():
    verify_files()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print("Cannot open"); sys.exit(1)

    w,h = int(cap.get(3)), int(cap.get(4))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_vid = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))
    csv_f = open(OUTPUT_CSV,'w',newline='',encoding='utf-8'); writer=csv.writer(csv_f)
    writer.writerow(["frame","timestamp","fps","num_tracks","track_ids"])

    metric  = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric, max_age=MAX_AGE, n_init=N_INIT)
    yolo    = YOLO(YOLO_WEIGHTS)

    prev_t, frame_idx = time.time(), 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        now = time.time()
        fps = 1.0/(now-prev_t) if now!=prev_t else fps_in
        prev_t = now

        preds = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
        results = preds[0] if preds else None
        bboxes, scores = [], []
        if results and hasattr(results,'boxes'):
            xyxy  = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), c in zip(xyxy, confs):
                if c<MIN_CONFIDENCE: continue
                bboxes.append([int(x1),int(y1),int(x2)-int(x1),int(y2)-int(y1)])
                scores.append(float(c))

        dets=[]
        for (x,y,w_,h_), conf in zip(bboxes,scores):
            crop = frame[y:y+h_, x:x+w_]
            if crop.size==0: continue
            hist = color_histogram(crop)
            dets.append(Detection([x,y,w_,h_], conf, hist))

        tracker.predict(); tracker.update(dets)

        active=[]
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update!=0: continue
            x1,y1,x2,y2 = tr.to_tlbr().astype(int)
            tid=tr.track_id; active.append(tid)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID_{tid}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.putText(frame, f"Frame: {frame_idx}", (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        ts = time.strftime("%H:%M:%S", time.localtime())
        writer.writerow([frame_idx,ts,f"{fps:.2f}",len(active),";".join(map(str,active))])
        out_vid.write(frame)
        cv2.imshow("Hist Only", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); out_vid.release(); csv_f.close(); cv2.destroyAllWindows()
    print(f"Saved {OUTPUT_VIDEO}, {OUTPUT_CSV}")

if __name__=="__main__":
    main()
