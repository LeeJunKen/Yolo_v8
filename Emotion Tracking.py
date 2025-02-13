# import os
# import cv2
# import numpy as np
# from keras.models import model_from_json
# from keras.preprocessing import image
# from tensorflow.python.keras.models import load_model
# from ultralytics import YOLO  # YOLOv8
# import matplotlib.pyplot as plt
# import subprocess # OpenFace (chạy qua subprocess hoặc gọi trực tiếp)
#
# # Load mô hình cảm xúc
# model =load_model("raf_model.h5")
#
# # Khởi tạo YOLOv8
# yolo_model = YOLO("runs/detect/train5/weights/best.pt")
#
# # Phát hiện khuôn mặt và thêm skeleton
# def detect_face_and_skeleton(frame):
#     results = yolo_model(frame)
#     for result in results:
#         for box in result.boxes.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             face = frame[y1:y2, x1:x2]
#
#             if face.size == 0:
#                 continue  # Bỏ qua nếu vùng khuôn mặt rỗng
#
#             face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#
#             # Lưu face_gray tạm thời để gọi OpenFace
#             temp_image_path = "temp_face.jpg"
#             cv2.imwrite(temp_image_path, face_gray)
#             landmarks = extract_landmarks(temp_image_path, "temp_output.txt")
#
#             # Vẽ skeleton lên khuôn mặt
#             for (x, y) in landmarks:
#                 cv2.circle(face, (int(x), int(y)), 2, (0, 255, 0), -1)
#
#             # Resize và chuẩn hóa khuôn mặt để đưa vào mô hình cảm xúc
#             face_resized = cv2.resize(face_gray, (48, 48))
#             mg_pixels = np.expand_dims(face_resized, axis=-1)  # Thêm chiều kênh (48, 48, 1)
#             img_pixels = np.expand_dims(img_pixels, axis=0)  # Thêm chiều batch (1, 48, 48, 1)
#             img_pixels /= 255.0
#
#             predictions = model.predict(img_pixels)
#             max_index = np.argmax(predictions[0])
#             emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#             predicted_emotion = emotions[max_index]
#
#             # Hiển thị kết quả
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, predicted_emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     return frame
#
# # Hàm trích xuất landmarks từ OpenFace
# def extract_landmarks(image_path, output_path):
#     command = f"FaceLandmarkImg.exe -f {image_path} -of {output_path}"
#     subprocess.run(command, shell=True)
#
#     landmarks = []
#     with open(output_path, "r") as file:
#         for line in file:
#             if line.startswith("landmark"):
#                 x, y = map(float, line.split()[1:3])
#                 landmarks.append((x, y))
#
#     return landmarks
#
# # Mở webcam và chạy phát hiện cảm xúc theo thời gian thực
# cap = cv2.VideoCapture(r"a.jpg")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_with_emotion = detect_face_and_skeleton(frame)
#     resized_frame = cv2.resize(frame_with_emotion, (1000, 700))
#     cv2.imshow('Emotion Detection with Skeleton', resized_frame)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()











import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model

#load model
model = load_model("raf_model.h5")


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
## detec face

cap=cv2.VideoCapture(0)   # bật webcam

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows

