import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
import motmetrics as mm
from scipy.optimize import linear_sum_assignment
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

# -------------------------------
# 1. Utility functions
# -------------------------------
def color_histogram(img, bins=(8,8,8)):
    """Compute normalized HSV color histogram."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def normalize_cost(mat):
    """Linearly normalize a cost matrix to [0,1]."""
    mn, mx = mat.min(), mat.max()
    return (mat - mn) / (mx - mn + 1e-6)

# def load_mot_gt(gt_path):
#     """
#     Load MOT-format ground truth from .txt:
#      frame, id, x, y, w, h, conf, class, visibility
#     Returns dict: frame_index -> [(id, [x1,y1,x2,y2]), ...]
#     """
#     df = pd.read_csv(gt_path, header=None)
#     df.columns = ["frame","id","x","y","w","h","conf","cls","vis"]
#     gt = {}
#     for _, row in df.iterrows():
#         f = int(row.frame)
#         x1, y1 = row.x, row.y
#         x2, y2 = x1 + row.w, y1 + row.h
#         gt.setdefault(f, []).append((int(row.id), [x1,y1,x2,y2]))
#     return gt

# -------------------------------
# 2. Initialize models & tracker
# -------------------------------
# YOLOv8 for detection
model = YOLO('runs/detect/train3/weights/best.pt')

# ORB for feature extraction
orb = cv2.ORB_create(nfeatures=256)

# Deep SORT tracker with infinite matching_threshold to disable its gating
metric = NearestNeighborDistanceMetric(
    "cosine",
    matching_threshold=float('inf'),
    budget=1000
)
# set n_init=1 so new tracks appear immediately
tracker = Tracker(metric, max_age=30, n_init=1)

# -------------------------------
# 3. Prepare video & ground truth
# -------------------------------
cap = cv2.VideoCapture(r"E:\DoAn\Data\test_class.mp4")
processing_width = 320

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'output_tracking.mp4', fourcc, 20.0,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

# gt_dict = load_mot_gt(r"E:\DoAn\Data\gt.txt")
# acc = mm.MOTAccumulator(auto_id=True)

# -------------------------------
# 4. Main processing loop
# -------------------------------
frame_idx = 0
fps_start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    scale_ratio = processing_width / W
    small = cv2.resize(frame, (processing_width, int(H * scale_ratio)))
    scale_back = W / processing_width

    # 4.1 Detect
    results = model.predict(small, conf=0.5, verbose=False)[0]
    dets, det_feats = [], []

    for box in results.boxes:
        x, y, w, h = map(int, box.xywh[0])
        x1, y1 = max(0, x - w//2), max(0, y - h//2)
        crop = small[y1:y1+h, x1:x1+w]
        if crop.size == 0:
            continue

        # ORB descriptor
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None:
            orb_feat = des.astype(np.float32).mean(axis=0)
            hist = color_histogram(crop)
            feat = np.hstack((orb_feat, hist))
        else:
            # fallback to histogram only
            feat = color_histogram(crop)

        dets.append([x1, y1, w, h])
        det_feats.append(feat)

    tracker.predict()
    # Chỉ lấy những track đã confirm và đã có feature
    confirmed_tracks = [
        t for t in tracker.tracks
        if t.is_confirmed() and len(t.features) > 0 and t.time_since_update <= 1
    ]

    # 4.3 Association via Hungarian on IOU + L2(feature)
    if confirmed_tracks and dets:
        trk_boxes = np.array([t.to_tlbr() for t in confirmed_tracks])
        # an toàn: mỗi t.features có ít nhất 1 phần tử
        trk_feats = np.vstack([t.features[-1] for t in confirmed_tracks])
        det_boxes = np.array(dets)
        det_feats = np.vstack(det_feats)

        # cost: IOU
        iou_cost = 1.0 - mm.distances.iou_matrix(trk_boxes, det_boxes)
        # cost: L2 on feature vector
        feat_cost = np.linalg.norm(trk_feats[:, None] - det_feats[None, :], axis=2)

        # normalize and combine
        iou_c = normalize_cost(iou_cost)
        feat_c = normalize_cost(feat_cost)
        cost_matrix = 0.5 * iou_c + 0.5 * feat_c

        # gating
        cost_matrix[cost_matrix > 0.7] = 1e6

        # Hungarian assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        matches = [(r, c) for r, c in zip(rows, cols) if cost_matrix[r, c] < 1e6]
        matched_t = {r for r, _ in matches}
        matched_d = {c for _, c in matches}
        unmatched_d = [j for j in range(len(dets)) if j not in matched_d]

        # build Detection list for tracker.update
        updated = []
        for r, c in matches:
            trk = confirmed_tracks[r]
            trk.features.append(det_feats[c])
            updated.append(Detection(dets[c], 1.0, det_feats[c]))
        for j in unmatched_d:
            updated.append(Detection(dets[j], 1.0, det_feats[j]))

        tracker.update(updated)
    else:
        # no valid matches
        new_dets = [Detection(dets[i], 1.0, det_feats[i]) for i in range(len(dets))]
        tracker.update(new_dets)

    # 4.4 Draw boxes & collect for metrics
    pred_boxes, pred_ids = [], []
    for trk in tracker.tracks:
        if not trk.is_confirmed() or trk.time_since_update > 1:
            continue
        x1_s, y1_s, x2_s, y2_s = trk.to_tlbr()
        x1 = int(x1_s * scale_back)
        y1 = int(y1_s * scale_back)
        x2 = int(x2_s * scale_back)
        y2 = int(y2_s * scale_back)
        pred_boxes.append([x1, y1, x2, y2])
        pred_ids.append(trk.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {trk.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4.5 Update MOT metrics
    # fno = frame_idx + 1
    # if fno in gt_dict:
    #     gt_entries = gt_dict[fno]
    #     gt_ids   = [e[0] for e in gt_entries]
    #     gt_boxes = [e[1] for e in gt_entries]
    # else:
    #     gt_ids, gt_boxes = [], []
    # D = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    # acc.update(gt_ids, pred_ids, D)

    # 4.6 Write frame, display, FPS
    out.write(frame)
    cv2.imshow("ORB+Histogram DeepSORT", frame)
    frame_count += 1
    if time.time() - fps_start >= 1.0:
        fps = frame_count / (time.time() - fps_start)
        fps_start = time.time()
        frame_count = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# -------------------------------
# 5. Cleanup & Reporting
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

# mh = mm.metrics.create()
# summary = mh.compute(
#     acc,
#     metrics=[
#         'num_false_positives','num_switches','num_fragmentations',
#         'idf1','precision','recall','mota','motp',
#         'mostly_tracked','mostly_lost'
#     ],
#     name='ORB_DeepSORT'
# )
# print(mm.io.render_summary(
#     summary,
#     formatters=mh.formatters,
#     namemap=mm.io.motchallenge_metric_names
# ))
