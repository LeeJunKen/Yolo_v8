#!/usr/bin/env python
import sys
import os
import time
import socket
import json
import threading
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
from YOLO import predict_emotion
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.modules['tensorflow'] = tf
np.int = int

from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# --- CẤU HÌNH ---
VIDEO_PATH    = r"E:\DoAn\Data\video_2.mp4"
GT_PATH       = r"Tracking\gt.txt"       # (có thể bỏ nếu chỉ cần gửi detections)
YOLO_WEIGHTS  = r"runs/detect/train_detection/weights/best.pt"
MIN_CONFIDENCE= 0.3
MAX_COSINE_DIST= 0.5
NN_BUDGET     = 50
MAX_AGE       = 9999
N_INIT        = 3
DETECT_INTERVAL = 3  # Mỗi 3 frame mới detect lại
TCP_HOST      = '127.0.0.1'
TCP_PORT      = 5000
# -----------------

def load_mot_gt(gt_path):
    df = pd.read_csv(gt_path, header=None)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    gt = {}
    for _, r in df.iterrows():
        f = int(r.frame)
        box = [r.x, r.y, r.x + r.w, r.y + r.h]
        gt.setdefault(f, []).append((int(r.id), box))
    return gt
class EmotionDetection(Detection):
    def __init__(self, tlwh, confidence, feature, emotion):
        super().__init__(tlwh, confidence, feature)
        self.emotion = emotion
def color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        bins, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Nếu cần GT metrics, giữ nguyên
gt_dict = load_mot_gt(GT_PATH)
acc = mm.MOTAccumulator(auto_id=True)

# Biến toàn cục để store các kết quả detections mới nhất
# Mỗi frame sẽ có danh sách detections: [(id, x, y, w, h), ...]
latest_results = {
    "frame": 0,
    "detections": []    # dạng list of dict: {"id": int,"x":int,"y":int,"w":int,"h":int}
}

# Khởi tạo socket server
def start_tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)
    print(f"[Python Service] Listening on {TCP_HOST}:{TCP_PORT} ...")
    conn, addr = server.accept()
    print(f"[Python Service] Client connected from {addr}")
    # Mỗi khi có client kết nối, ta bật luồng gửi JSON
    sender_thread = threading.Thread(target=send_loop, args=(conn,))
    sender_thread.daemon = True
    sender_thread.start()

# Vòng lặp gửi JSON liên tục, 30–60 FPS tùy máy
def send_loop(conn):
    global latest_results
    while True:
        try:
            data = json.dumps(latest_results)
            # Đảm bảo mỗi JSON kết thúc bằng ký tự newline để C# dễ phân tách
            message = data + "\n"
            conn.sendall(message.encode('utf-8'))
            # Mình giả sử cứ mỗi 1/30 giây gửi, bạn có thể điều chỉnh tuỳ thực tế
            time.sleep(1.0 / 60.0)
        except Exception as e:
            print(f"[Python Service] Error sending data: {e}")
            break
    conn.close()
    print("[Python Service] Client disconnected.")

def main_tracking_loop():
    # Mở video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video"); sys.exit(1)

    # Thiết lập YOLO + Deep SORT
    yolo = YOLO(YOLO_WEIGHTS)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DIST, NN_BUDGET)
    tracker = Tracker(metric, max_age=MAX_AGE, n_init=N_INIT)

    prev_t = time.time()
    frame_idx = 0
    dets_cache = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Chỉ detect mỗi DETECT_INTERVAL frame
        if frame_idx % DETECT_INTERVAL == 0:
            preds = yolo(frame, conf=MIN_CONFIDENCE, verbose=False)
            results = preds[0] if preds else None
            bboxes, scores = [], []
            if results and hasattr(results, 'boxes'):
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    if c < MIN_CONFIDENCE:
                        continue
                    bboxes.append([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)])
                    scores.append(float(c))

            dets = []
            for (x, y, w_, h_), conf in zip(bboxes, scores):
                crop = frame[y:y + h_, x:x + w_]
                if crop.size == 0:
                    continue
                emo, _ = predict_emotion(crop)
                hist = color_histogram(crop)
                dets.append(EmotionDetection([x,y,w_,h_], conf, hist, emo))
            dets_cache = np.array(dets)

        tracker.predict()
        tracker.update(dets_cache)

        for det, tr in zip(dets_cache, tracker.tracks):
            tr.emotion = det.emotion

        # Chuẩn bị kết quả detections để gửi cho C#
        current_list = []
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update != 0:
                continue
            x1, y1, x2, y2 = tr.to_tlbr().astype(int)
            w_ = x2 - x1
            h_ = y2 - y1
            emo = getattr(tr, 'emotion', 'Unknown')
            current_list.append({
                "id": int(tr.track_id),
                "x": int(x1),
                "y": int(y1),
                "w": int(w_),
                "h": int(h_),
                "emotion": emo
            })

        # Cập nhật latest_results (dùng chung cho luồng send_loop đọc)
        latest_results["frame"] = frame_idx
        latest_results["detections"] = current_list

        # (Nếu bạn muốn hiển thị lên console để debug, có thể uncomment)
        # print(f"[Python Service] Frame {frame_idx}: {current_list}")

    cap.release()
    print("[Python Service] Video ended.")


if __name__ == "__main__":
    # 1) Bắt đầu TCP server (đợi C# kết nối)
    server_thread = threading.Thread(target=start_tcp_server)
    server_thread.daemon = True
    server_thread.start()

    # 2) Chạy vòng lặp tracking chính
    main_tracking_loop()
    # Khi video xong, bạn có thể dừng server hoặc giữ cho client tự disconnect
    print("[Python Service] Exiting.")
