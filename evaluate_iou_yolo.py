import cv2
import os
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def yolo_to_bbox(yolo_line, img_w, img_h):
    class_id, x, y, w, h = map(float, yolo_line.strip().split())
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return [x1, y1, x2, y2]

if __name__ == "__main__":
    image_name = "frame_00000"
    image_path = f"frames_output/{image_name}.jpg"
    gt_label_path = r"F:\TaiLieuDoAn\Yolo_v8\test\labels\frame_00000_jpg.rf.8ae05e5f8774e530f2efabd6e379dacc.txt"
    pred_label_path = f"yolo_preds/predict2/labels/{image_name}.txt"

    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        exit()

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(gt_label_path, "r") as f:
        gt_boxes = [yolo_to_bbox(line, w, h) for line in f if line.strip()]

    with open(pred_label_path, "r") as f:
        pred_boxes = [yolo_to_bbox(line, w, h) for line in f if line.strip()]

    print(f"🖼 Đánh giá ảnh: {image_name}.jpg")
    best_ious = []

    for idx, pred in enumerate(pred_boxes):
        ious = [compute_iou(pred, gt) for gt in gt_boxes]
        if ious:
            max_iou = max(ious)
            best_ious.append(max_iou)
            print(f"IoU cao nhất cho box {idx+1}: {max_iou:.4f}")
        else:
            max_iou = 0

        # Vẽ box dự đoán
        x1, y1, x2, y2 = pred
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Ghi giá trị IoU lên box
        label = f"box: {idx + 1}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for gt in gt_boxes:
        x1, y1, x2, y2 = gt
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Ground Truth: Xanh

    if best_ious:
        print(f"IoU trung bình toàn ảnh: {np.mean(best_ious):.4f}")
    else:
        print("Không có box dự đoán hoặc nhãn ground truth.")

    # Hiển thị ảnh
    cv2.imshow("Ground Truth (Blue) vs Prediction (Red)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
