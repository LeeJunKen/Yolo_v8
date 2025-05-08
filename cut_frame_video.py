import cv2
import os

def extract_frames(video_path, output_dir, every_n_frames=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không mở được video, hãy kiểm tra đường dẫn hoặc định dạng.")
        return
    frame_id = 0
    saved_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % every_n_frames == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_id:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()
    print(f"Đã lưu {saved_id} frame vào thư mục {output_dir}")

# Ví dụ sử dụng
extract_frames(r"video\dct.mp4", "frames_output", every_n_frames=1)
