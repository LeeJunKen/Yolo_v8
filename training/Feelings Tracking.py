import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets, transforms
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import load_model
# 1. Hàm tiền xử lý dữ liệu
def process_data(train_dir, test_dir):
    """
    Hàm xử lý và chuẩn bị dữ liệu từ thư mục.

    Args:
    - train_dir (str): Đường dẫn đến thư mục train.
    - test_dir (str): Đường dẫn đến thư mục test.

    Returns:
    - X_train, y_train: Dữ liệu và nhãn huấn luyện.
    - X_test, y_test: Dữ liệu và nhãn kiểm tra.
    """
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dữ liệu từ thư mục
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Chuyển dữ liệu thành numpy arrays
    X_train = torch.stack([x[0] for x in train_data], dim=0).numpy()
    y_train = np.array([x[1] for x in train_data])

    X_test = torch.stack([x[0] for x in test_data], dim=0).numpy()
    y_test = np.array([x[1] for x in test_data])

    # Reshape dữ liệu để phù hợp với mô hình Keras
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    # One-hot encoding nhãn cảm xúc
    num_labels = 7
    y_train = to_categorical(y_train, num_classes=num_labels)
    y_test = to_categorical(y_test, num_classes=num_labels)

    return X_train, y_train, X_test, y_test, num_labels


# 2. Hàm xây dựng mô hình
def build_model(input_shape, num_labels):
    """
    Xây dựng mô hình CNN.

    Args:
    - input_shape (tuple): Kích thước đầu vào (48, 48, 1).
    - num_labels (int): Số lượng class cảm xúc.

    Returns:
    - model: Mô hình CNN đã được xây dựng.
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model


# 3. Hàm huấn luyện mô hình
def train_model(model, X_train, y_train, X_test, y_test, batch_size=64, epochs=30):
    """
    Huấn luyện mô hình CNN.

    Args:
    - model: Mô hình đã được xây dựng.
    - X_train, y_train: Dữ liệu và nhãn huấn luyện.
    - X_test, y_test: Dữ liệu và nhãn kiểm tra.
    - batch_size (int): Kích thước batch (mặc định là 64).
    - epochs (int): Số epoch huấn luyện (mặc định là 30).

    Returns:
    - model: Mô hình đã được huấn luyện.
    """
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              shuffle=True)
    model.save("raf_model.h5")
    print("Mô hình đã được lưu thành công!")
    return model


# Hàm dự đoán cảm xúc từ ảnh khuôn mặt
def predict_emotion(model, face_img, class_names):
    """
    Dự đoán cảm xúc từ ảnh khuôn mặt.

    Args:
    - model: Mô hình CNN đã huấn luyện.
    - face_img: Ảnh khuôn mặt đã được tiền xử lý (grayscale, kích thước 48x48).
    - class_names: Danh sách các class cảm xúc.

    Returns:
    - predicted_emotion: Tên cảm xúc được dự đoán.
    - confidence: Xác suất của cảm xúc dự đoán.
    """
    face_resized = cv2.resize(face_img, (48, 48))  # Resize ảnh về 48x48
    img_pixels = image.img_to_array(face_resized)
    img_pixels = np.expand_dims(img_pixels, axis=0)  # Thêm chiều batch
    img_pixels /= 255.0  # Chuẩn hóa giá trị pixel

    # Dự đoán cảm xúc
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    predicted_emotion = class_names[max_index]
    confidence = predictions[0][max_index]

    return predicted_emotion, confidence


# 4. Main function để chạy toàn bộ
if __name__ == "__main__":
    # train_dir = r"RAF-DB\train"  # Đường dẫn đến dữ liệu train
    # test_dir = r"RAF-DB\test"  # Đường dẫn đến dữ liệu test
    #
    # # Xử lý dữ liệu
    # X_train, y_train, X_test, y_test, num_labels = process_data(train_dir, test_dir)
    #
    # # Xây dựng mô hình
    # model = build_model(input_shape=(48, 48, 1), num_labels=num_labels)
    #
    # # Huấn luyện mô hình
    # train_model(model, X_train, y_train, X_test, y_test)

    # Tải lại mô hình đã huấn luyện
    model = load_model("raf_model.h5")
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Đọc ảnh đầu vào để dự đoán cảm xúc
    face_img = cv2.imread(r"a.jpg", cv2.IMREAD_GRAYSCALE)
    if face_img is None:
        print("Không đọc được ảnh. Vui lòng kiểm tra đường dẫn.")
    # Kiểm tra và gọi hàm predict
    predicted_emotion, confidence = predict_emotion(model, face_img, class_names)
    print(f"Dự đoán cảm xúc: {predicted_emotion} (Độ tin cậy: {confidence:.2f})")
    cv2.imshow('Ảnh',face_img)
    cv2.waitKey(0)  # Chờ phím bất kỳ để đóng cửa sổ
    cv2.destroyAllWindows()
