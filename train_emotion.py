from multiprocessing import freeze_support
import yaml
from ultralytics import YOLO

def main():
    model=YOLO("yolo11n-cls.pt")
    results = model.train(
        data="F:\TaiLieuDoAn\Yolo_v8\RAF-DB",    # dict from YAML
        epochs=50,         # number of epochs
        imgsz=224,         # image size for classification
        batch=32,          # batch size
        device=0,          # use GPU 0
        task="classify"    # classification task
    )

    # 4. Print out where weights are saved
    print("Training completed.")
    print("Results saved to:", results.path)
    print("Best model weights:", results.best)

if __name__ == "__main__":
    freeze_support()
    main()

