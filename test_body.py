import cv2
import mediapipe as mp
import pandas as pd
# Đọc ảnh từ webcam
cap = cv2.VideoCapture(r"E:\DoAn\Data\6313772697550.mp4")

mpPose = mp.solutions.pose
pose = mpPose. Pose()
mpDraw = mp.solutions.drawing_utils
lm_list = []

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)

    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
def draw_landmark_on_image (mpDraw,results, img):
    mpDraw.draw_landmarks (img, results.pose_landmarks, mpPose. POSE_CONNECTIONS)
    for id, lm in enumerate (results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Nhân diễn pose
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        # Ghi nhận thông số khung xương
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        # Vẽ khung xương lên ảnh
        frame = draw_landmark_on_image(mpDraw, results, frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows ()