import cv2
from ultralytics import YOLO
import time

def draw_boxes(image, boxes, scores, labels, conf_threshold=0.25):
    """
    image: NumPy array dạng BGR
    boxes: shape (N, 4)
    scores: shape (N,)
    labels: shape (N,)
    class_names: dict hoặc list map từ label index -> tên lớp
    conf_threshold: ngưỡng để hiển thị box
    """
    for (box, score, label) in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Gắn nhãn
        # cls_name = class_names[int(label)] if int(label) in class_names else str(label)
        text = f"Person: {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


model = YOLO("runs/detect/train5/weights/best.pt")  # Đường dẫn tới trọng số đã huấn luyện
# results = model.predict(
#         source="SCUT_HEAD_Part_A\\JPEGImages\\PartA_00000.jpg",
#         save=False,
#         imgsz=640,
#         conf=0.5
#     )
#
# print(results)
    # Dự đoán trên dữ liệu test

start_time = time.time()
frame_count = 0
cap = cv2.VideoCapture(r"E:\DoAn\Data\test.mp4")
fps=0
# #
while True:
    ret, frame = cap.read()

    if not ret:
        break
    results = model.predict(
        source=frame,
        save=False,
        imgsz=640,
        conf=0.5
    )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    frame_count += 1

    # Tính FPS cứ mỗi 1 giây
    elapsed = time.time() - start_time
    if elapsed >= 1.0:  # đủ 1 giây
        fps = frame_count / elapsed
        # print(f"FPS: {fps:.2f}")

        # Reset
        frame_count = 0
        start_time = time.time()
    frame_out = draw_boxes(frame, boxes, scores, labels, conf_threshold=0.3)
    cv2.putText(frame, f"FPS: {round(fps,3)}", (10, height-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#
cap.release()
cv2.destroyAllWindows()