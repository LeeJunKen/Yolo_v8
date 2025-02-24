import cv2
from ultralytics import YOLO
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load model YOLO
model = YOLO("runs/detect/train5/weights/best.pt")
cap = cv2.VideoCapture("video/test.mp4")


def calculate_histogram(face_img, h_bins=50, s_bins=60):
    # Chuyển ảnh sang không gian màu HSV
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    # Tính histogram cho kênh H và S (bỏ kênh V nếu muốn)
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def histogram_distance(hist1, hist2):
    # Tính khoảng cách giữa 2 histogram bằng phương pháp chi-square
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)


def compute_cost_with_histogram(tracker_hist, detection_face_img, weight_hist=1.0):
    # Tính histogram cho khuôn mặt mới và kết hợp với trọng số (nếu cần)
    detection_hist = calculate_histogram(detection_face_img)
    hist_cost = histogram_distance(tracker_hist, detection_hist)
    return weight_hist * hist_cost


def detect_faces_yolo(frame):
    """
    Sử dụng YOLO để phát hiện khuôn mặt trong một frame.
    Trả về danh sách bounding boxes [(x1, y1, x2, y2, conf)].
    """
    results = model.predict(source=frame, conf=0.3, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


# Khai báo biến lưu trữ histogram và ID của frame trước
previous_histograms = []
previous_ids = []
next_id = 0


def assign_ids_with_histogram(previous_histograms, current_histograms, cost_threshold=100.0):
    """
    Sử dụng Hungarian Algorithm để ghép đôi các detection của frame hiện tại với tracker từ frame trước,
    dựa trên chi phí histogram (chi-square distance).

    Nếu chi phí nhỏ hơn ngưỡng cho phép thì gán cùng ID, ngược lại gán ID mới.
    """
    global next_id
    # Nếu không có tracker từ frame trước, gán ID mới cho tất cả detection hiện tại
    if len(previous_histograms) == 0:
        ids = list(range(next_id, next_id + len(current_histograms)))
        next_id += len(current_histograms)
        return ids

    num_prev = len(previous_histograms)
    num_curr = len(current_histograms)
    cost_matrix = np.zeros((num_prev, num_curr))

    # Tạo ma trận chi phí dựa trên khoảng cách histogram
    for i, prev_hist in enumerate(previous_histograms):
        for j, curr_hist in enumerate(current_histograms):
            cost_matrix[i, j] = histogram_distance(prev_hist, curr_hist)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Ban đầu, gán -1 cho các detection hiện tại (chưa được ghép đôi)
    assigned_ids = [-1] * num_curr

    # Ghép đôi các tracker với detection nếu chi phí dưới ngưỡng
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            assigned_ids[j] = previous_ids[i]

    # Gán ID mới cho các detection chưa được ghép đôi
    for j in range(num_curr):
        if assigned_ids[j] == -1:
            assigned_ids[j] = next_id
            next_id += 1

    return assigned_ids


def process_frame(frame):
    global previous_histograms, previous_ids
    faces = detect_faces_yolo(frame)
    current_histograms = []
    boxes = []

    # Với mỗi khuôn mặt được phát hiện, tính histogram
    for (x1, y1, x2, y2, conf) in faces:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        hist = calculate_histogram(face_crop)
        current_histograms.append(hist)
        boxes.append((x1, y1, x2, y2, conf))

    # Gán ID dựa trên việc so sánh histogram của frame trước và hiện tại
    ids = assign_ids_with_histogram(previous_histograms, current_histograms, cost_threshold=100.0)

    # Cập nhật dữ liệu cho frame sau
    previous_histograms = current_histograms
    previous_ids = ids
    return list(zip(ids, boxes))


# Vòng lặp xử lý video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = process_frame(frame)

    # Vẽ bounding box và ID lên frame
    for face_id, (x1, y1, x2, y2, conf) in results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
