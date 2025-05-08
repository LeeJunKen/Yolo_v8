import cv2

import numpy as np
import time
import motmetrics as mm
import pandas as pd
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

from YOLO import *

# Load YOLOv8 model và Deep SORT extractor

orb = cv2.ORB_create(nfeatures=256)
# Khởi tạo Deep SORT Tracker
metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.5)
tracker = Tracker(metric, max_age=30)

class EmotionDetection(Detection):
    def __init__(self, tlwh, confidence, feature, emotion):
        super().__init__(tlwh, confidence, feature)
        self.emotion = emotion


def color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
def orb_embedding(face_img):
    """
    face_img: ảnh BGR chứa khuôn mặt đã crop
    Trả về: vector embedding cố định (float32)
    """
    # Chuyển sang gray
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Phát hiện keypoints + tính descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors)==0:
        # Không có descriptor, có thể trả về vector zeros
        return np.zeros(orb.descriptorSize(), dtype=np.float32)

    # Pooling: mean của tất cả descriptors
    # Chuyển về float32 để chính xác hơn
    descriptors = descriptors.astype(np.float32)
    embedding = descriptors.mean(axis=0)  # shape = (32,)

    return embedding
def load_mot_gt(gt_path):
    """
    Hàm load dữ liệu Ground Truth của MOT từ file txt.
    Định dạng dự kiến mỗi dòng:
    frame, id, x, y, w, h, conf, class, visibility
    Trả về: dictionary với key là frame number và value là danh sách (id, [x1, y1, x2, y2])
    """
    df = pd.read_csv(gt_path, header=None)
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    gt_dict = {}
    for _, row in df.iterrows():
        frame_num = int(row["frame"])
        # Chuyển đổi bounding box từ (x, y, w, h) sang (x1, y1, x2, y2)
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["w"]
        y2 = y1 + row["h"]
        if frame_num not in gt_dict:
            gt_dict[frame_num] = []
        gt_dict[frame_num].append((int(row["id"]), [x1, y1, x2, y2]))
    return gt_dict

# Load GT từ MOT dataset (update đường dẫn cho phù hợp)
gt_dict = load_mot_gt(r"E:\DoAn\Data\Tracking_1\gt.txt")

# r"E:\DoAn\Data\test_class.mp4"
cap = cv2.VideoCapture(r"E:\DoAn\Data\Tracking_1\video.mp4")
processing_width = 320

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracking.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

fps_start_time = time.time()
frame_count = 0
video_start_time = time.time()
# Thiết lập motmetrics accumulator
acc = mm.MOTAccumulator(auto_id=True)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    scale_ratio = processing_width / width
    frame_resized = cv2.resize(frame, (processing_width, int(height * scale_ratio)))

    results = predict_face(frame_resized, conf=0.5, verbose=False)


    detections = []
    pred_boxes, pred_ids = [], []

    # Xử lý các detection từ YOLO
    for bbox in results.boxes:
        x, y, w, h = map(int, bbox.xywh[0])
        # Chuyển tọa độ từ tâm sang góc trên-trái
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        face_img = frame_resized[y1:y1 + h, x1:x1 + w]



        if face_img.size == 0:
            continue
        emotion = predict_emotion(face_img)

        face_img_resized = cv2.resize(face_img, (64, 128))
        embedding_cnn = orb_embedding(face_img_resized)

        embedding_hist = color_histogram(face_img_resized)
        combined_embedding = np.hstack((embedding_cnn,embedding_hist))
        det = EmotionDetection([x1, y1, w, h], float(bbox.conf), combined_embedding, emotion[0])
        detections.append(det)


    tracker.predict()
    tracker.update(detections)

    for det, track in zip(detections, tracker.tracks):
        track.emotion = det.emotion

    # Tỷ lệ scale từ frame đã resize về kích thước ban đầu
    scale_ratio_back = width / processing_width

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        emotion = track.emotion
        x1, y1, x2, y2 = map(int, track.to_tlbr() * scale_ratio_back)
        pred_boxes.append([x1, y1, x2, y2])
        pred_ids.append(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}_{emotion}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính FPS
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = time.time()

    # Ghi frame vào video kết quả


    # Cập nhật MOT metrics cho frame hiện tại
    current_frame = frame_idx + 1  # GT frame thường bắt đầu từ 1
    if current_frame in gt_dict:
        gt_data = gt_dict[current_frame]
        gt_ids = [entry[0] for entry in gt_data]
        gt_boxes = [entry[1] for entry in gt_data]
        print(gt_boxes)
        for box in gt_boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 250), 2)
    else:
        gt_ids = []
        gt_boxes = []

    distance_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    acc.update(gt_ids, pred_ids, distance_matrix)

    frame_idx += 1

    cv2.imshow('YOLOv8 + Deep SORT Face Tracking', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
video_end_time = time.time()

# Tính tổng thời gian và FPS trung bình
total_time = video_end_time - video_start_time
avg_fps = frame_idx / total_time if total_time > 0 else 0.0

# Tính toán và hiển thị các chỉ số tracking
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_false_positives','num_switches','num_fragmentations','idf1','precision','recall','mota','motp','mostly_tracked','mostly_lost'], name='YOLO_DeepSORT')
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

print(f"Processed {frame_idx} frames in {total_time:.2f} seconds. Average FPS: {avg_fps:.2f}")