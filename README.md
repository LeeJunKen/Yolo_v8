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

## 🧑‍💻 Công nghệ sử dụng
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- Python 3.10  
- OpenCV, NumPy  
- Deep SORT (tracking-by-detection)  
- Hungarian Algorithm  
- CUDA (GPU tăng tốc)

## 📊 Dữ liệu
- **SCUT-HEAD** – phát hiện & theo dõi khuôn mặt (~111k khuôn mặt).  
- **RAF-DB** – phân loại cảm xúc (4 lớp cơ bản: Happiness, Sadness, Surprise, Neutral).

## ⚙️ Cấu hình huấn luyện
- **YOLOv8 (detection)**  
  - Epochs: 100  
  - Image size: 640  
  - Batch: 16  
- **YOLOv8 (classification)**  
  - Epochs: 50  
  - Image size: 224  
  - Batch: 32  

## 🚀 Kết quả
- **Phát hiện khuôn mặt:** mAP@0.5 = 0.942  
- **Theo dõi (Deep SORT):** MOTA = 96.1%, IDF1 = 96.8%  
- **Nhận diện cảm xúc:** Top-1 Accuracy ≈ 91%  
- **Tốc độ xử lý:** ~13 FPS

## 📂 Cấu trúc thư mục
```bash
Yolo_v8/
│── data/              # Bộ dữ liệu huấn luyện & test
│── models/            # Pretrained & trained weights
│── src/               # Mã nguồn chính
│   ├── detection/     # YOLOv8 detection
│   ├── tracking/      # Deep SORT + Hungarian Matching
│   └── emotion/       # Emotion recognition
│── results/           # Kết quả thực nghiệm
│── requirements.txt   # Các thư viện cần thiết
│── README.md          # Giới thiệu dự án


