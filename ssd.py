import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt


# ======= PHẦN 1: Dataset chuẩn theo VOC (label: person) =======
class SCUTHeadDataset(Dataset):
    def __init__(self, root, image_set='train'):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.anno_dir = os.path.join(root, "Annotations")
        list_path = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")

        with open(list_path) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.transforms = Compose([
            Resize((300, 300)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, image_id + ".jpg")
        xml_path = os.path.join(self.anno_dir, image_id + ".xml")

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.ids))

        boxes, labels = self.parse_voc_xml(xml_path)
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.ids))

        image = self.transforms(image)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

    def parse_voc_xml(self, xml_path):
        boxes, labels = [], []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name != "person":
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)
        except Exception as e:
            print(f"Lỗi đọc {xml_path}: {e}")
        return boxes, labels

# ======= PHẦN 2: Huấn luyện SSD300 =======
def collate_fn(batch):
    return tuple(zip(*batch))

def train_ssd_model():
    root_dir = "SCUT_HEAD_Part_A_&_B_1"

    print("Load dữ liệu huấn luyện...")
    train_dataset = SCUTHeadDataset(root=root_dir, image_set='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssd300_vgg16(pretrained=False, num_classes=2).to(device)  # 0: background, 1: person

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    print("Bắt đầu huấn luyện SSD300...")
    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                print(f"NaN tại batch {i} → bỏ qua.")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"📘 Epoch [{epoch}/30], Loss: {total_loss:.4f}")

        # (Tùy chọn) Lưu mô hình
        torch.save(model.state_dict(), f"ssd_epoch_{epoch}.pth")

    print("Huấn luyện hoàn tất!")



def predict_and_visualize(image_path, model_path="ssd_epoch_30.pth", threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ảnh gốc
    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    # Resize + normalize giống khi train
    transform = Compose([
        Resize((300, 300)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, C, H, W)

    # Load model
    model = ssd300_vgg16(pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dự đoán
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    # Vẽ kết quả trên ảnh gốc
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image)
    ax = plt.gca()

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold and label == 1:
            xmin, ymin, xmax, ymax = box.cpu().tolist()
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"person: {score:.2f}", color='red',
                    fontsize=8, backgroundcolor='blue')

    plt.axis("off")
    plt.title(f"Predictions (threshold = {threshold})")
    plt.show()

# ======= CHẠY =======

if __name__ == "__main__":
    # train_ssd_model()
    predict_and_visualize("SCUT_HEAD_Part_A_&_B_1\JPEGImages\PartA_00000.jpg", "ssd_epoch_30.pth", threshold=0.5)
