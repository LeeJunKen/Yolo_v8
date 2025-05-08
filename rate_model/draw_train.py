# import matplotlib.pyplot as plt
# import os
# from pathlib import Path
#
# # === 1. Lấy thống kê split ===
# train_img_dir = Path("SCUT_HEAD_DATASET_1/images/train")
# val_img_dir   = Path("SCUT_HEAD_DATASET_1/images/test")
#
# n_train_imgs = len(list(train_img_dir.glob("*.jpg")))
# n_val_imgs   = len(list(val_img_dir.glob("*.jpg")))
#
# # === 2. Lấy thống kê số bbox (head) ===
# train_lbl_dir = Path("SCUT_HEAD_DATASET_1/labels/train")
# val_lbl_dir   = Path("SCUT_HEAD_DATASET_1/labels/test")
#
# def count_boxes(label_dir):
#     total = 0
#     for file in label_dir.glob("*.txt"):
#         with open(file, "r") as f:
#             total += sum(1 for _ in f if _.strip())
#     return total
#
# n_train_boxes = count_boxes(train_lbl_dir)
# n_val_boxes   = count_boxes(val_lbl_dir)
#
# # === 3. Vẽ 2 biểu đồ song song ===
# fig, axes = plt.subplots(1, 2, figsize=(10,4))
#
# # (a) Split ảnh
# axes[0].bar(["Train", "Val"], [n_train_imgs, n_val_imgs], color=["navy", "brown"])
# for idx,val in enumerate([n_train_imgs, n_val_imgs]):
#     axes[0].text(idx, val+5, f"{val}", ha="center", fontsize=9)
# axes[0].set_ylim(0, max(n_train_imgs, n_val_imgs)*1.15)
# axes[0].set_title("SCUT‑HEAD dataset split")
# axes[0].set_ylabel("Số lượng ảnh")
#
# # (b) Phân bố bbox
# axes[1].bar(["Head (train)", "Head (val)"],
#             [n_train_boxes, n_val_boxes],
#             color=["steelblue", "seagreen"])
# for idx,val in enumerate([n_train_boxes, n_val_boxes]):
#     axes[1].text(idx, val+5, f"{val}", ha="center", fontsize=9)
# axes[1].set_ylim(0, max(n_train_boxes, n_val_boxes)*1.15)
# axes[1].set_title("Phân bố head trong nhãn")
#
# plt.tight_layout()
# plt.savefig("dataset_overview.png", dpi=300)
# plt.show()
#
# "Đã tạo biểu đồ dataset_overview.png"





# draw_stats_parta.py
# import os
# import xml.etree.ElementTree as ET
# from pathlib import Path
# import matplotlib.pyplot as plt
#
# # ----- 1. Cấu hình đường dẫn gốc -----
# ROOT = Path(r"SCUT_HEAD_Part_B")         # <‑‑ chỉnh đường dẫn nếu đặt khác
# SPLIT_FILES = {
#     "Train": "train.txt",
#     "Val"  : "val.txt"
# }
# # Nếu bạn muốn test:
# # SPLIT_FILES["Test"] = "test.txt"
#
# # ----- 2. Hàm đọc ID ảnh từ ImageSets/Main/*.txt -----
# def load_ids(split_name, txt_file):
#     file_path = ROOT / "ImageSets" / "Main" / txt_file
#     if not file_path.exists():
#         print(f"⚠️  Không tìm thấy {file_path}")
#         return []
#     with open(file_path, encoding="utf‑8") as f:
#         return [ln.strip() for ln in f if ln.strip()]
#
# # ----- 3. Hàm đếm bbox (head) trong Annotation XML -----
# def count_boxes(ids):
#     total = 0
#     for img_id in ids:
#         xml_path = ROOT / "Annotations" / f"{img_id}.xml"
#         if not xml_path.exists():
#             print(f"⚠️  Thiếu annotation: {xml_path}")
#             continue
#         tree = ET.parse(xml_path)
#         total += len(tree.findall(".//object"))   # mỗi <object> = 1 bbox
#     return total
#
# # ----- 4. Thu thập thống kê -----
# img_counts  = {}    # {split: số ảnh}
# bbox_counts = {}    # {split: số bbox}
#
# for split, txt_name in SPLIT_FILES.items():
#     ids                 = load_ids(split, txt_name)
#     img_counts[split]   = len(ids)
#     bbox_counts[split]  = count_boxes(ids)
#
# # ----- 5. Vẽ biểu đồ song song -----
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#
# # (a) Split ảnh
# splits         = list(img_counts.keys())
# img_values     = [img_counts[s] for s in splits]
# bars1          = axes[0].bar(splits, img_values)
# axes[0].set_title("SCUT‑HEAD dataset split")
# axes[0].set_ylabel("Số lượng ảnh")
# for bar in bars1:
#     h = int(bar.get_height())
#     axes[0].text(bar.get_x() + bar.get_width()/2, h + 5, f"{h}",
#                  ha="center", va="bottom", fontsize=9)
#
# # (b) Phân bố head (bbox)
# bbox_values = [bbox_counts[s] for s in splits]
# labels2     = [f"Head ({s.lower()})" for s in splits]
# bars2       = axes[1].bar(labels2, bbox_values)
# axes[1].set_title("Phân bố head trong nhãn")
# axes[1].set_ylabel("Số lượng head")
# for bar in bars2:
#     h = int(bar.get_height())
#     axes[1].text(bar.get_x() + bar.get_width()/2, h + 5, f"{h}",
#                  ha="center", va="bottom", fontsize=9)
#
# plt.tight_layout()
# plt.savefig("dataset_overview_parta.png", dpi=300)   # lưu hình nếu cần
# plt.show()

# import os
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # import pandas as pd
# #
# # # Thư mục dữ liệu
# # train_root = Path("RAF-DB/train")
# # test_root = Path("RAF-DB/test")
# #
# # # Lấy danh sách lớp (folder) và sắp xếp theo số
# # classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()], key=lambda x: int(x))
# #
# # # Đếm số lượng ảnh trong từng lớp
# # train_counts = [len(list((train_root / cls).glob("*.*"))) for cls in classes]
# # test_counts  = [len(list((test_root  / cls).glob("*.*"))) for cls in classes]
# #
# # # Tạo DataFrame và hiển thị
# # df = pd.DataFrame({
# #     "Lớp": classes,
# #     "Train": train_counts,
# #     "Test": test_counts
# # })
# #
# # # Vẽ biểu đồ so sánh
# # x = list(range(1, len(classes)+1))
# # width = 0.35
# #
# # fig, ax = plt.subplots(figsize=(10,6))
# # bars1 = ax.bar([i - width/2 for i in x], train_counts, width, label='Train', color='tab:blue')
# # bars2 = ax.bar([i + width/2 for i in x], test_counts,  width, label='Test',  color='tab:orange')
# #
# # ax.set_xlabel("Lớp cảm xúc")
# # ax.set_ylabel("Số lượng ảnh")
# # ax.set_title("So sánh phân bố ảnh Train và Test theo lớp cảm xúc")
# # ax.set_xticks(x)
# # ax.set_xticklabels(x)
# # ax.legend()
# #
# # # Ghi số lên đầu mỗi bar
# # for bar in bars1 + bars2:
# #     h = bar.get_height()
# #     ax.annotate(f"{h}",
# #                 xy=(bar.get_x() + bar.get_width()/2, h),
# #                 xytext=(0,3), textcoords="offset points",
# #                 ha='center', va='bottom')
# #
# # plt.tight_layout()
# # plt.savefig("dataset_raf.png", dpi=300)
# # plt.show()

from pathlib import Path
from collections import defaultdict

def count_raf_images(root_dir: str,
                     splits=("train", "test"),
                     img_exts=(".jpg")):
    """
    Trả về:
      • dict tổng ảnh cho từng split   → {'train': 12271, 'test': 3068}
      • dict chi tiết từng class/split → {'train': {'0': 2000, ...}, 'test': {...}}
    """
    root          = Path(root_dir)
    total_counts  = {}
    class_counts  = defaultdict(dict)

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"⚠️  Không tìm thấy thư mục {split_dir}")
            total_counts[split] = 0
            continue

        total = 0
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            n_imgs = sum(1 for p in cls_dir.iterdir()
                         if p.suffix.lower() in img_exts)
            class_counts[split][cls_dir.name] = n_imgs
            total += n_imgs

        total_counts[split] = total

    return total_counts, class_counts


# -------- Thí dụ sử dụng --------
if __name__ == "__main__":
    raf_root = r"F:\TaiLieuDoAn\Yolo_v8\RAF-DB"      # chỉnh đường dẫn cho phù hợp

    totals, details = count_raf_images(raf_root)
    print("Tổng ảnh:", totals)
    print("\nChi tiết theo lớp:")
    for split in details:
        print(f"  {split}")
        for cls, n in details[split].items():
            print(f"    class {cls}: {n}")
