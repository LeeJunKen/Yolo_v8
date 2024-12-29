import cv2
import torch
import os
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import shutil

def Preprocess_data():
    # Đường dẫn đến train.txt của Part A và Part B
    train_a_path = r"SCUT_HEAD_Part_A\ImageSets\Main\train.txt"
    train_b_path = r"SCUT_HEAD_Part_B\ImageSets\Main\train.txt"

    # Đọc danh sách ảnh
    with open(train_a_path, "r") as f:
        train_a_images = f.readlines()
    with open(train_b_path, "r") as f:
        train_b_images = f.readlines()

    # Gộp danh sách
    all_train_images = train_a_images + train_b_images
    all_train_images = list(set(all_train_images))  # Loại bỏ trùng lặp
    all_train_images.sort()  # Sắp xếp

    # Đường dẫn thư mục cần tạo
    output_train_dir = "SCUT_HEAD_Part_A_&_B\ImageSets\Main"
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_train_dir, exist_ok=True)
    # Lưu vào file mới
    output_train_path = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\train.txt"
    with open(output_train_path, "w") as f:
        f.writelines(all_train_images)
    print(f"Gộp xong tập train tại: {output_train_path}")

    # Thư mục Part A và Part B
    images_a_dir = "SCUT_HEAD_Part_A\JPEGImages"
    images_b_dir = "SCUT_HEAD_Part_B\JPEGImages"
    annotations_a_dir = "SCUT_HEAD_Part_A\Annotations"
    annotations_b_dir = "SCUT_HEAD_Part_B\Annotations"

    # Thư mục chung
    combined_images_dir = "SCUT_HEAD_Part_A_&_B\JPEGImages"
    combined_annotations_dir = "SCUT_HEAD_Part_A_&_B\Annotations"
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(combined_annotations_dir, exist_ok=True)

    # Gộp ảnh
    for img_file in os.listdir(images_a_dir):
        shutil.copy(os.path.join(images_a_dir, img_file), combined_images_dir)

    for img_file in os.listdir(images_b_dir):
        shutil.copy(os.path.join(images_b_dir, img_file), combined_images_dir)

    # Gộp annotation
    for xml_file in os.listdir(annotations_a_dir):
        shutil.copy(os.path.join(annotations_a_dir, xml_file), combined_annotations_dir)

    for xml_file in os.listdir(annotations_b_dir):
        shutil.copy(os.path.join(annotations_b_dir, xml_file), combined_annotations_dir)

    print("Gộp ảnh và annotation hoàn tất!")


    # Đường dẫn đến val.txt của Part A và Part B
    val_a_path = r"SCUT_HEAD_Part_A\ImageSets\Main\val.txt"
    val_b_path = r"SCUT_HEAD_Part_B\ImageSets\Main\val.txt"
    # Đường dẫn thư mục val.txt
    output_val_dir = "SCUT_HEAD_Part_A_&_B\ImageSets\Main"
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_val_dir, exist_ok=True)
    # Đọc danh sách ảnh
    with open(val_a_path, "r") as f:
        val_a_images = f.readlines()

    with open(val_b_path, "r") as f:
        val_b_images = f.readlines()

    # Gộp danh sách
    all_val_images = val_a_images + val_b_images
    all_val_images = list(set(all_val_images))  # Loại bỏ trùng lặp
    all_val_images.sort()  # Sắp xếp

    # Lưu vào file mới
    output_val_path = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\val.txt"
    with open(output_val_path, "w") as f:
        f.writelines(all_val_images)

    print(f"Gộp xong tập val tại: {output_val_path}")

    def convert_xml_to_yolo(xml_dir, output_dir, classes):
        os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

        for xml_file in os.listdir(xml_dir):
            if not xml_file.endswith(".xml"):
                continue

            # Đọc file XML
            anno_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(anno_path)
            root = tree.getroot()

            # Lấy kích thước ảnh
            size = root.find("size")
            if size is None:
                print(f"File XML thiếu thông tin kích thước: {xml_file}")
                continue

            try:
                img_width = int(size.find("width").text)
                img_height = int(size.find("height").text)
                if img_width == 0 or img_height == 0:
                    print(f"Kích thước ảnh không hợp lệ trong file XML: {xml_file} (width={img_width}, height={img_height})")
                    continue
            except (AttributeError, ValueError):
                print(f"Kích thước ảnh không hợp lệ trong file XML: {xml_file}")
                continue

            # Tên file TXT tương ứng
            txt_filename = os.path.splitext(xml_file)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, "w") as f:
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    if class_name not in classes:
                        continue  # Bỏ qua nếu class không thuộc danh sách classes
                    class_id = classes.index(class_name)

                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # Chuyển đổi sang YOLO format
                    try:
                        x_center = ((xmin + xmax) / 2) / img_width
                        y_center = ((ymin + ymax) / 2) / img_height
                        box_width = (xmax - xmin) / img_width
                        box_height = (ymax - ymin) / img_height
                    except ZeroDivisionError:
                        print(f"ZeroDivisionError tại file: {xml_file}, bỏ qua đối tượng này.")
                        continue

                    # Ghi vào file TXT
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        print(f"Đã chuyển đổi XML sang YOLO TXT và lưu tại: {output_dir}")
    train = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\train.txt"
    val = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\val.txt"
    # Đường dẫn tới thư mục XML và thư mục lưu TXT
    annotation = "SCUT_HEAD_Part_A_&_B\Annotations"
    output_dir = "SCUT_HEAD_Part_A_&_B\labels"
    classes = ["person"]  # Danh sách các lớp (ví dụ: chỉ có lớp 'person')
    # Chuyển đổi
    convert_xml_to_yolo(annotation, output_dir, classes)
    # Nội dung file data.yaml
    data_yaml_content = """
    train: /content/SCUT_HEAD_DATASET/images/train
    val: /content/SCUT_HEAD_DATASET/images/val
    
    nc: 1
    names: ['person']
    """



    # Danh sách các thư mục cần tạo
    dirs = [
        r"SCUT_HEAD_DATASET\images\train",
        r"SCUT_HEAD_DATASET\images\val",
        r"SCUT_HEAD_DATASET\labels\train",
        r"SCUT_HEAD_DATASET\labels\val"
    ]

    # Tạo các thư mục nếu chưa tồn tại
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Đã kiểm tra hoặc tạo thư mục: {dir_path}")


    # Đường dẫn dữ liệu gốc
    image_dir = r"SCUT_HEAD_Part_A_&_B\JPEGImages"
    label_dir = r"SCUT_HEAD_Part_A_&_B\labels"

    # Đường dẫn thư mục đích
    output_image_train = r"SCUT_HEAD_DATASET\images\train"
    output_image_val = r"SCUT_HEAD_DATASET\images\val"
    output_label_train = r"SCUT_HEAD_DATASET\labels\train"
    output_label_val = r"SCUT_HEAD_DATASET\labels\val"

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_image_train, exist_ok=True)
    os.makedirs(output_image_val, exist_ok=True)
    os.makedirs(output_label_train, exist_ok=True)
    os.makedirs(output_label_val, exist_ok=True)

    # Danh sách file train và val
    train_txt = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\train.txt"
    val_txt = r"SCUT_HEAD_Part_A_&_B\ImageSets\Main\val.txt"

    # Hàm xử lý việc di chuyển file
    def move_files(file_list, image_output_dir, label_output_dir):
        missing_files = []  # Danh sách file bị thiếu
        for file in file_list:
            try:
                # Di chuyển ảnh
                shutil.copy(os.path.join(image_dir, file + ".jpg"), image_output_dir)
                # Di chuyển nhãn
                shutil.copy(os.path.join(label_dir, file + ".txt"), label_output_dir)
            except FileNotFoundError as e:
                print(f"File không tồn tại: {e.filename}")
                missing_files.append(file)

        return missing_files

    # Đọc danh sách file train
    with open(train_txt, "r") as f:
        train_files = [line.strip() for line in f.readlines()]

    # Đọc danh sách file val
    with open(val_txt, "r") as f:
        val_files = [line.strip() for line in f.readlines()]

    # Di chuyển file train
    missing_train_files = move_files(train_files, output_image_train, output_label_train)

    # Di chuyển file val
    missing_val_files = move_files(val_files, output_image_val, output_label_val)

    # In danh sách file bị thiếu
    if missing_train_files:
        print("\nCác file thiếu trong tập train:")
        for file in missing_train_files:
            print(file)

    if missing_val_files:
        print("\nCác file thiếu trong tập val:")
        for file in missing_val_files:
            print(file)

    print("\nHoàn thành di chuyển dữ liệu!")

    # Loại bỏ file bị thiếu khỏi train.txt
    with open(train_txt, "w") as f:
        f.writelines([file + "\n" for file in train_files if file not in missing_train_files])

    # Loại bỏ file bị thiếu khỏi val.txt
    with open(val_txt, "w") as f:
        f.writelines([file + "\n" for file in val_files if file not in missing_val_files])

def check_dataset_split(train_txt, val_txt, image_dir, label_dir):
    """
    Kiểm tra số lượng ảnh và nhãn trong tập train và val.

    Args:
        train_txt (str): Đường dẫn tới file train.txt.
        val_txt (str): Đường dẫn tới file val.txt.
        image_dir (str): Thư mục chứa ảnh gốc.
        label_dir (str): Thư mục chứa nhãn gốc.

    Returns:
        None. In ra thông tin về số lượng tập train và val.
    """
    # Đọc danh sách file từ train.txt
    with open(train_txt, "r") as f:
        train_files = [line.strip() for line in f.readlines()]

    # Đọc danh sách file từ val.txt
    with open(val_txt, "r") as f:
        val_files = [line.strip() for line in f.readlines()]

    # Kiểm tra sự tồn tại của file ảnh và nhãn trong tập train
    missing_train_images = []
    missing_train_labels = []
    for file in train_files:
        img_path = os.path.join(image_dir, file + ".jpg")
        label_path = os.path.join(label_dir, file + ".txt")
        if not os.path.exists(img_path):
            missing_train_images.append(file)
        if not os.path.exists(label_path):
            missing_train_labels.append(file)

    # Kiểm tra sự tồn tại của file ảnh và nhãn trong tập val
    missing_val_images = []
    missing_val_labels = []
    for file in val_files:
        img_path = os.path.join(image_dir, file + ".jpg")
        label_path = os.path.join(label_dir, file + ".txt")
        if not os.path.exists(img_path):
            missing_val_images.append(file)
        if not os.path.exists(label_path):
            missing_val_labels.append(file)

    # Kết quả
    print(f"Số lượng file trong train.txt: {len(train_files)}")
    print(f"Số lượng file trong val.txt: {len(val_files)}")
    print(f"Số lượng file ảnh thiếu trong tập train: {len(missing_train_images)}")
    print(f"Số lượng file nhãn thiếu trong tập train: {len(missing_train_labels)}")
    print(f"Số lượng file ảnh thiếu trong tập val: {len(missing_val_images)}")
    print(f"Số lượng file nhãn thiếu trong tập val: {len(missing_val_labels)}")

    # In danh sách file thiếu nếu có
    if missing_train_images:
        print("\nDanh sách ảnh bị thiếu trong tập train:")
        for file in missing_train_images:
            print(file)

    if missing_train_labels:
        print("\nDanh sách nhãn bị thiếu trong tập train:")
        for file in missing_train_labels:
            print(file)

    if missing_val_images:
        print("\nDanh sách ảnh bị thiếu trong tập val:")
        for file in missing_val_images:
            print(file)

    if missing_val_labels:
        print("\nDanh sách nhãn bị thiếu trong tập val:")
        for file in missing_val_labels:
            print(file)

    print("\nKiểm tra hoàn tất!")



def main():
    # Load mô hình đã huấn luyện
    model = YOLO("runs/detect/train5/weights/best.pt")  # Đường dẫn tới trọng số đã huấn luyện

    # Dự đoán trên dữ liệu test
    results = model.predict(
        source="SCUT_HEAD_Part_A\\JPEGImages\\PartA_00000.jpg",  # Thư mục chứa ảnh test
        save=False,  # Lưu kết quả
        imgsz=640,  # Kích thước ảnh
        conf=0.5  # Ngưỡng confidence
    )
    print("Dự đoán hoàn tất. Kết quả đã được lưu trong thư mục 'runs/detect/predict'")

    for r in results:
        img = r.orig_img  # Lấy ảnh gốc từ kết quả YOLO
        # Lấy bounding boxes
        boxes = r.boxes.xyxy.cpu().numpy()  # Chuyển bounding boxes sang numpy
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box[:4])  # Lấy tọa độ bounding box
            # Vẽ bounding box lên ảnh
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='g', facecolor='none'))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Hiển thị ảnh
        plt.axis('off')
        plt.show()
def train():
    # Khởi tạo mô hình YOLOv8
    model = YOLO("yolov8n.pt")  # Có thể thay bằng yolov8s.pt, yolov8m.pt tùy tài nguyên

    # Huấn luyện mô hình
    model.train(
        data="data.yaml",  # Đường dẫn tới file data.yaml
        epochs=50,  # Số epoch
        imgsz=640,  # Kích thước ảnh
        batch=16,  # Batch size
        device="0"  # GPU hoặc CPU
    )

    # Kiểm tra mô hình sau khi huấn luyện
    metrics = model.val(data="data.yaml", imgsz=640)

if __name__ == "__main__":
    # train_txt = r"F:\TaiLieuDoAn\Yolo_v8\SCUT_HEAD_Part_A_&_B\ImageSets\Main\train.txt"
    # val_txt = r"F:\TaiLieuDoAn\Yolo_v8\SCUT_HEAD_Part_A_&_B\ImageSets\Main\val.txt"
    # image_dir = r"F:\TaiLieuDoAn\Yolo_v8\SCUT_HEAD_Part_A_&_B\JPEGImages"
    # label_dir = r"F:\TaiLieuDoAn\Yolo_v8\SCUT_HEAD_Part_A_&_B\labels"
    # check_dataset_split(train_txt, val_txt, image_dir, label_dir)
    main()



