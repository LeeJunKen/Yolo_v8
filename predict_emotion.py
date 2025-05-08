from ultralytics import YOLO
import cv2
from pathlib import Path

# 1. Định nghĩa list tên lớp theo đúng thứ tự folder (0→6)
CLASS_NAMES = [
    "Surprise",   # 0
    "Fear",       # 1
    "Disgust",    # 2
    "Happiness",  # 3
    "Sadness",    # 4
    "Anger",      # 5
    "Neutral"     # 6
]

# 2. Load model classification đã huấn luyện
cls_model = YOLO("runs/classify/train8/weights/best.pt")

def predict_emotion(image_input, model=cls_model, imgsz=224, device=0):
    """
    Dự đoán cảm xúc, trả về (idx, label, confidence)
    idx: số lớp (0-6)
    label: tên lớp tương ứng
    confidence: xác suất
    """
    # Đọc ảnh
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {image_input}")
    else:
        img = image_input

    # Predict
    results = model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
    if not results:
        raise RuntimeError("Model không trả về kết quả.")
    r = results[0]

    # Lấy index và confidence từ Probs
    idx = int(r.probs.top1)         # ví dụ 3
    conf = float(r.probs.top1conf)  # ví dụ 0.87

    # Map sang nhãn
    label = CLASS_NAMES[idx]        # ví dụ "Happiness"

    return idx, label, conf

# Ví dụ sử dụng
if __name__ == "__main__":
    test_img = r"F:\TaiLieuDoAn\Yolo_v8\RAF-DB\test\1\test_0002_aligned.jpg"
    idx, emotion, conf = predict_emotion(test_img)
    print(f"Class index: {idx}")
    print(f"Cảm xúc: {emotion} ({conf:.2%})")
