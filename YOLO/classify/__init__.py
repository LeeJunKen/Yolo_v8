from ultralytics import YOLO
model = YOLO('runs/classify/train_emotion_cus/weights/best.pt')
def predict_emotion(image, imgsz=224, device=0):
    """
    Dự đoán cảm xúc, trả về (idx, label, confidence)
    idx: số lớp (0-6)
    label: tên lớp tương ứng
    confidence: xác suất
    """
    CLASS_NAMES = [
        "Surprise",  # 0
        "Happiness",  # 3
        "Sadness",  # 4
    ]

    # Predict
    results = model.predict(source=image, imgsz=imgsz, verbose=False)
    if not results:
        raise RuntimeError("Model không trả về kết quả.")
    r = results[0]

    # Lấy index và confidence từ Probs
    idx = int(r.probs.top1)         # ví dụ 3
    conf = float(r.probs.top1conf)  # ví dụ 0.87

    # Map sang nhãn
    label = CLASS_NAMES[idx]        # ví dụ "Happiness"

    return label, conf