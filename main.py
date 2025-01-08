from deepface import DeepFace
import cv2
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import numpy as np


model = YOLO("runs/detect/train5/weights/best.pt")

def calculate_distance(embedding1, embedding2):
    """
    Tính khoảng cách cosine giữa 2 embedding.
    Khoảng cách càng nhỏ thì 2 embedding càng giống nhau.
    """
    return cosine(embedding1, embedding2)
def hungarian_algorithm(cost_matrix):
    """
    Thực hiện Hungarian Algorithm trên ma trận chi phí.
    cost_matrix: Ma trận chi phí (NxM) giữa các đối tượng ở frame trước và frame hiện tại.
    Trả về: Danh sách các cặp (index trước, index hiện tại).
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))
def detect_faces_yolo(frame):
    """
    Sử dụng YOLO để phát hiện khuôn mặt trong một frame.
    Trả về danh sách bounding boxes [(x1, y1, x2, y2, conf)].
    """
    results = model.predict(source=frame, conf=0.3, verbose=False)  # Confidence threshold = 0.3
    boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))

    return boxes
def extract_embedding_deepface(face_img, model_name="VGG-Face"):
    """
    Sử dụng DeepFace để trích xuất embedding từ khuôn mặt.
    face_img: Ảnh khuôn mặt đã được cắt từ bounding box.
    model_name: Tên mô hình DeepFace (default: VGG-Face).
    """
    embedding = DeepFace.represent(face_img, model_name=model_name, enforce_detection=False)[0]['embedding']
    return embedding

previous_embeddings = []
previous_ids = []
def assign_ids(previous_embeddings, current_embeddings, threshold=0.6):
    """
    Gán ID duy nhất cho các khuôn mặt dựa trên Hungarian Algorithm.
    previous_embeddings: Embedding từ frame trước.
    current_embeddings: Embedding từ frame hiện tại.
    threshold: Ngưỡng khoảng cách để xác định 'cùng một khuôn mặt'.
    """
    ids = []
    unmatched_ids = set(range(len(previous_embeddings)))  # ID chưa khớp từ frame trước

    if len(previous_embeddings) == 0:
        # Nếu là frame đầu tiên, gán ID theo thứ tự
        return list(range(len(current_embeddings))), set(range(len(current_embeddings)))

    # Tạo ma trận chi phí dựa trên khoảng cách cosine giữa các embedding
    cost_matrix = np.zeros((len(previous_embeddings), len(current_embeddings)))

    for i, prev_emb in enumerate(previous_embeddings):
        for j, curr_emb in enumerate(current_embeddings):
            cost_matrix[i, j] = calculate_distance(prev_emb, curr_emb)

    # Hungarian Algorithm để tìm cách so khớp tối ưu
    matches = hungarian_algorithm(cost_matrix)

    assigned_ids = {}
    for prev_idx, curr_idx in matches:
        if cost_matrix[prev_idx, curr_idx] <= threshold:
            assigned_ids[curr_idx] = prev_idx  # ID khớp
            unmatched_ids.discard(prev_idx)

    # Gán ID mới cho các khuôn mặt không khớp
    next_id = max(assigned_ids.values(), default=-1) + 1
    for i in range(len(current_embeddings)):
        if i not in assigned_ids:
            assigned_ids[i] = next_id
            next_id += 1

    return [assigned_ids[i] for i in range(len(current_embeddings))], unmatched_ids
def process_frame_with_hungarian(frame):
    """
    Tích hợp YOLO + DeepFace + Hungarian Algorithm để theo dõi khuôn mặt.
    Trả về bounding box, ID, và embedding.
    """
    global previous_embeddings, previous_ids

    # Bước 1: Phát hiện bounding box của khuôn mặt bằng YOLO
    faces = detect_faces_yolo(frame)

    # Bước 2: Trích xuất embedding của các bounding box bằng DeepFace
    current_embeddings = []
    face_boxes = []
    for (x1, y1, x2, y2, conf) in faces:
        face_crop = frame[y1:y2, x1:x2]
        embedding = extract_embedding_deepface(face_crop, model_name="VGG-Face")
        current_embeddings.append(embedding)
        face_boxes.append((x1, y1, x2, y2, conf))

    # Bước 3: So khớp với frame trước bằng Hungarian Algorithm
    current_ids, unmatched_ids = assign_ids(previous_embeddings, current_embeddings, threshold=0.6)

    # Cập nhật embedding và ID của frame hiện tại
    previous_embeddings = current_embeddings
    previous_ids = current_ids

    return list(zip(current_ids, face_boxes))

cap = cv2.VideoCapture("video/test.mp4")  # Đọc video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý từng frame
    results = process_frame_with_hungarian(frame)

    # Hiển thị bounding box và ID
    for face_id, (x1, y1, x2, y2, conf) in results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {face_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()