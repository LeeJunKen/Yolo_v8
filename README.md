# YOLOv8 - Classroom Monitoring System 🎓📹

Ứng dụng YOLO trong bài toán giám sát lớp học.  
Dự án được phát triển trong khuôn khổ khóa luận tốt nghiệp ngành Công nghệ Thông tin – Trường Đại học Sài Gòn (05/2025).

## 📌 Giới thiệu
Hệ thống giám sát lớp học theo thời gian thực với 3 chức năng chính:
- **Phát hiện khuôn mặt** bằng YOLOv8.
- **Theo dõi và gán ID ổn định** cho từng học sinh bằng Deep SORT + Hungarian Matching.
- **Nhận diện cảm xúc** (Happiness, Sadness, Surprise, Neutral) từ khuôn mặt đã phát hiện.

Ứng dụng hỗ trợ:
- Điểm danh tự động.
- Theo dõi trạng thái học sinh (tập trung, mệt mỏi, ngạc nhiên…).
- Phân tích hành vi và cảm xúc để cải thiện phương pháp giảng dạy.


<div align="center">
  
![Introduce.](assets/introduce.jpg)  

*Hình: Giới thiệu khóa luận.*

</div>

## 🧑‍💻 Công nghệ sử dụng
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- Python 3.10  
- OpenCV, NumPy  
- Deep SORT (tracking-by-detection)  
- Hungarian Algorithm  
- CUDA (GPU tăng tốc)

## 📊 Dữ liệu
- **SCUT-HEAD** – phát hiện & theo dõi khuôn mặt (~111k khuôn mặt).

<div align="center">
  
![Thống kê phân chia tập SCUT-HEAD và phân bố nhãn head trong train/val.](rate_model/dataset_overview.png)  

*Hình: Thống kê phân chia tập SCUT-HEAD và phân bố nhãn head trong train/val.*

</div>


- **RAF-DB** – phân loại cảm xúc (4 lớp cơ bản: Happiness, Sadness, Surprise, Neutral).

<div align="center">
  
![So sánh phân bố tập dữ liệu RAF.](rate_model/dataset_raf.png)  

*Hình: So sánh phân bố tập dữ liệu RAF.*

</div>


## 🛠️ Cài đặt và chạy chương trình

### 1. Clone repo
```bash
git clone https://github.com/LeeJunKen/Yolo_v8.git
cd Yolo_v8
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Chạy chương trình

#### 3.1. Demo
```bash
python demo.py
```

#### 3.2. Deep Sort - Trích xuất đặc trưng bằng Color Histogram và CNN
```bash
python use_DeepSORT.py
```

#### 3.3. Deep Sort - Trích xuất đặc trưng bằng Color Histogram
```bash
python use_DeepSORT_CH.py
```

#### 3.4. Deep Sort - Trích xuất đặc trưng bằng CNN
```bash
python use_DeepSORT_CNN.py
```

## ⚙️ Cấu hình huấn luyện
- **YOLOv8 (detection)**  
  - Epochs: 100  
  - Image size: 640  
  - Batch: 16  

<div align="center">
  
![Loss Scut-Head.](rate_model/loss_over_epoch.png)  

*Hình: Loss Over Epoch Scut-Head.*

</div>

- **YOLOv8 (classification)**  
  - Epochs: 50  
  - Image size: 224  
  - Batch: 32  

<div align="center">
  
![Loss RAF.](rate_model/loss_raf.png)  

*Hình: Loss Over Epoch RAF.*

</div>

## 🚀 Kết quả
- **Phát hiện khuôn mặt:** mAP@0.5 = 0.942  
- **Theo dõi (Deep SORT):** MOTA = 96.1%, IDF1 = 96.8%  
- **Nhận diện cảm xúc:** Top-1 Accuracy ≈ 91%  
- **Tốc độ xử lý:** ~30 FPS (RTX 5060)

<div align="center">

![Kết quả.](assets/result.png)  

*Hình: Kết quả được trích xuất trong video nhận diện và theo dõi.*

</div>


## 📂 Cấu trúc thư mục
```bash
Yolo_v8/
│── deep_sort/             # Thuật toán Deep Sort
│── Tracking/              # File MOT và Video kiểm thử
│── runs/                  # Pretrained & trained weights
│── results/               # Kết quả thực nghiệm
│── demo.py                # Ứng dụng demo về nhận diện và theo dõi
│── use_DeepSORT.py        # Trích xuất đặc trưng bằng Color Histogram và CNN
│── use_DeepSORT_CH.py     # Trích xuất đặc trưng bằng Color Histogram
│── use_DeepSORT_CNN.py    # Trích xuất đặc trưng bằng CNN
│── use_SORT.py            # Áp dụng thuật toán SORT
│── requirements.txt       # Các thư viện cần thiết
│── README.md              # Giới thiệu dự án

```
## 📖 Hướng phát triển
- Tích hợp multi-camera.
- Nâng cao tốc độ để đạt ≥ 25 FPS trên CPU.
- Mở rộng phân loại cảm xúc với nhiều trạng thái hơn.
- Kết hợp phân tích hành vi học sinh dựa trên pose estimation.

## 👨‍🎓 Tác giả
**Khóa luận tốt nghiệp – Trường Đại học Sài Gòn, 2025.**
- Tống Đức Duy
- Lê Trung Kiên


## 📚 Tài liệu tham khảo
- YOLO: [Redmon et al., 2016]
- Deep SORT: [Wojke et al., 2017]
- SCUT-HEAD Dataset
- RAF-DB Dataset

