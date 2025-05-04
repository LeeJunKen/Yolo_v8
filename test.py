from ultralytics import YOLO

# 1. Load model đã huấn luyện
model = YOLO('runs/classify/train8/weights/best.pt')

# 2. Chạy inference trên ảnh mới
results = model(r"E:\Dataset GT.v1i.yolov8-obb\train\images\test_class_mp4-0011_jpg.rf.abf68ae73d16a5c880b4db4ecca9326e.jpg", conf=0.25)[0]  # lấy kết quả của ảnh đầu tiên

# 3. In thông tin bbox, class và confidence
for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
    x1, y1, x2, y2 = box.tolist()
    class_id = int(cls)
    class_name = model.names[class_id]
    print(f"{class_name:7s} | conf={conf:.2f} | bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

# 4. (Tuỳ chọn) Lưu ảnh đã vẽ bbox & nhãn
# results.save(save_dir='runs/detect/emotion_results')
