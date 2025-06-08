

import sys
import cv2
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QSizePolicy, QPushButton
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

from ultralytics import YOLO
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

class VideoTracker:
    def __init__(self):
        self.yolo = YOLO("runs/detect/train_detection/weights/best.pt")
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 50)
        self.tracker = Tracker(metric, max_age=9999, n_init=3)
        self.cap = cv2.VideoCapture(r"Tracking\video.mp4")
        if not self.cap.isOpened():
            raise RuntimeError("Không thể mở video")
        self.prev_t = time.time()
        self.start_t = self.prev_t
        self.frame_idx = 0

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_idx += 1
        now = time.time()
        inst_fps = 1.0 / (now - self.prev_t) if now != self.prev_t else 0.0
        self.prev_t = now
        avg_fps = self.frame_idx / max(1e-6, now - self.start_t)

        if self.frame_idx % 3 == 0:
            results = self.yolo(frame, conf=0.3, verbose=False)[0]
            dets = []
            for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                hist = self.color_histogram(crop)
                dets.append(Detection([x1, y1, x2 - x1, y2 - y1], float(conf), hist))
            self.tracker.predict()
            self.tracker.update(np.array(dets))

        faces = []
        vis = frame.copy()
        for tr in self.tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update:
                continue
            x1, y1, x2, y2 = tr.to_tlbr().astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID: {tr.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            crop = frame[y1:y2, x1:x2]
            emotion = "Neutral"  # TODO: thay bằng model emotion thật
            faces.append((crop, tr.track_id, emotion))

        cv2.putText(vis, f"Inst FPS: {inst_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Avg FPS: {avg_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis, faces

    @staticmethod
    def color_histogram(img, bins=(8, 8, 8)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo")
        self.tracker = VideoTracker()

        root = QWidget()
        main_layout = QHBoxLayout(root)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.video_label)

        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Tạm dừng")
        self.pause_btn.clicked.connect(self.pause)
        btn_layout.addWidget(self.pause_btn)
        self.resume_btn = QPushButton("Tiếp tục")
        self.resume_btn.clicked.connect(self.resume)
        btn_layout.addWidget(self.resume_btn)
        left_layout.addLayout(btn_layout)
        main_layout.addWidget(left_widget, 2)

        # Bên phải: scroll danh sách face crops
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.faces_widget = QWidget()
        self.faces_layout = QVBoxLayout(self.faces_widget)
        self.faces_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.faces_widget)
        main_layout.addWidget(scroll, 1)

        self.setCentralWidget(root)

        # Timer ~30 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def pause(self):
        self.timer.stop()

    def resume(self):
        if not self.timer.isActive():
            self.timer.start(30)

    def update_frame(self):
        data = self.tracker.next_frame()
        if data is None:
            self.timer.stop()
            return
        frame, faces = data

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        for i in reversed(range(self.faces_layout.count())):
            wdg = self.faces_layout.takeAt(i).widget()
            if wdg:
                wdg.deleteLater()

        for crop, tid, emo in faces:
            if crop.size == 0:
                continue
            face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            fh, fw, fc = face_rgb.shape
            bpl = fc * fw
            qim = QImage(face_rgb.data, fw, fh, bpl, QImage.Format.Format_RGB888).copy()
            pix_face = QPixmap.fromImage(qim).scaledToWidth(100, Qt.TransformationMode.SmoothTransformation)

            lbl_img = QLabel()
            lbl_img.setPixmap(pix_face)

            info_box = QWidget()
            info_layout = QVBoxLayout(info_box)
            info_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            lbl_id = QLabel(f"ID: {tid}")
            lbl_emo = QLabel(f"Cảm xúc: {emo}")
            info_layout.addWidget(lbl_id)
            info_layout.addWidget(lbl_emo)

            box = QWidget()
            hb = QHBoxLayout(box)
            hb.setAlignment(Qt.AlignmentFlag.AlignLeft)
            hb.addWidget(lbl_img)
            hb.addWidget(info_box)
            self.faces_layout.addWidget(box)

        self.faces_widget.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())
