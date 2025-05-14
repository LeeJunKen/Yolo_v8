from ultralytics import YOLO
model = YOLO('runs/detect/train_detection/weights/best.pt')
def predict_face(frame, conf=0.5, verbose=False):
    return model.predict(frame, conf=0.5, verbose=False)[0]
