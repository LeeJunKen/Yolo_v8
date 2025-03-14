import scipy
import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment


# Hàm phụ trợ: tính vector đặc trưng từ bounding box
def bbox_feature(bbox):
    """
    Tính vector đặc trưng từ bbox.
    bbox: [x, y, w, h]
    Trả về vector [cx, cy, w, h], trong đó cx, cy là tọa độ tâm.
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


class HungarianMatching(object):
    def __init__(self):
        # feature_map: key -> {'key': key, 'features': list(feature vectors),
        # 'images': list(image patches), 'bboxes': list(bbox), 'active': bool, 'num': count}
        self.feature_map = {}
        self.feature_idx = 0
        self.feature_size = 15
        self.infinity = 100000.0
        # Ngưỡng so sánh; với vector từ bbox, bạn có thể điều chỉnh lại threshold cho phù hợp
        self.threshold = 0.3
        # Loại bỏ dependency vào VGG; thay vào đó, dùng hàm bbox_feature
        self.descriptor = None

    def start(self, image, bboxes):
        self.feature_map = {}
        self.feature_idx = 0
        key_bboxes = []
        for bbox in bboxes:
            # bbox ở định dạng [x, y, w, h]
            # Thay vì gọi self.descriptor.predict, ta tính feature từ bbox
            feature = bbox_feature(bbox)
            fvecs = [feature]  # lưu dưới dạng danh sách
            # Lấy image patch theo bbox
            image_bbox = image[int(bbox[1]):int(bbox[1] + bbox[3]),
                         int(bbox[0]):int(bbox[0] + bbox[2])]
            self.feature_map[self.feature_idx] = {
                'key': self.feature_idx,
                'features': fvecs,
                'images': [image_bbox],
                'bboxes': [bbox],
                'active': True,
                'num': 1
            }
            key_bboxes.append(self.feature_idx)
            self.feature_idx += 1
        return key_bboxes

    # end start

    def update(self, image, bboxes):
        # feature_map_keys: các key hiện có trong feature_map
        feature_map_keys = list(self.feature_map.keys())
        distance_matrix_feature_map = []  # ma trận khoảng cách giữa detection mới và các track cũ
        cur_features = []
        cur_images_box = []
        cur_batch_fvecs = []

        for idx, bbox in enumerate(bboxes):
            # Tính feature từ bbox (thay vì gọi descriptor.predict)
            feature = bbox_feature(bbox)
            # Để giữ định dạng như cũ (mảng các vector), ta lưu trong list
            batch_fvecs = [feature]
            # Lấy ảnh khuôn mặt từ frame dựa trên bbox
            image_bbox = image[int(bbox[1]):int(bbox[1] + bbox[3]),
                         int(bbox[0]):int(bbox[0] + bbox[2])]
            cur_images_box.append(image_bbox)

            cur_features.append(feature)
            cur_batch_fvecs.append(batch_fvecs)

            # Tính khoảng cách giữa đặc trưng mới và mỗi track cũ
            feature_map_distances = distance_face(self.feature_map, feature, self.threshold, self.infinity)

            # Xây dựng một hàng trong ma trận chi phí: mỗi hàng ứng với detection mới
            distance_feature_map = []
            for i in range(len(feature_map_keys)):
                distance_feature_map.append(feature_map_distances[feature_map_keys[i]])
            # Nếu số detection mới lớn hơn số track hiện có, mở rộng hàng với giá trị infinity
            for i in range(len(bboxes) - len(feature_map_keys)):
                distance_feature_map.append(self.infinity)
            distance_matrix_feature_map.append(distance_feature_map)
        # for

        # Hungarian Algorithm
        if len(bboxes) > 0:
            row_ind, col_ind = linear_sum_assignment(distance_matrix_feature_map)

        # Đánh dấu tất cả track cũ là không active
        for key in feature_map_keys:
            self.feature_map[key]["active"] = False

        # Ghép nối (matching)
        key_bboxes = []
        num_features = len(feature_map_keys)
        for idx, bbox in enumerate(bboxes):
            # Nếu detection mới được ghép nối với một track cũ (với khoảng cách nhỏ)
            if col_ind[idx] < num_features and distance_matrix_feature_map[idx][col_ind[idx]] < self.infinity:
                matched_key = feature_map_keys[col_ind[idx]]
                fvecs = self.feature_map[matched_key]["features"]
                # Nếu số mẫu vượt quá feature_size, loại bỏ mẫu cũ nhất
                if self.feature_map[matched_key]["num"] + 1 > self.feature_size:
                    fvecs = np.delete(fvecs, 0, axis=0)
                    self.feature_map[matched_key]["num"] -= 1
                    self.feature_map[matched_key]["images"].pop(0)
                    self.feature_map[matched_key]["bboxes"].pop(0)
                # Cập nhật track bằng cách thêm các vector đặc trưng mới
                fvecs = np.append(fvecs, cur_batch_fvecs[idx], axis=0)
                self.feature_map[matched_key]["features"] = fvecs
                self.feature_map[matched_key]["images"].append(cur_images_box[idx])
                self.feature_map[matched_key]["bboxes"].append(bbox)
                self.feature_map[matched_key]["active"] = True
                self.feature_map[matched_key]["num"] += 1
                key_bboxes.append(matched_key)
            else:  # Nếu detection không khớp với track nào, tạo track mới
                self.feature_map[self.feature_idx] = {
                    'key': self.feature_idx,
                    'features': cur_batch_fvecs[idx],
                    'images': [cur_images_box[idx]],
                    'bboxes': [bbox],
                    'active': True,
                    'num': 1
                }
                key_bboxes.append(self.feature_idx)
                self.feature_idx += 1
        # for
        return key_bboxes

    # end update

    def draw_bboxes(self, image):
        for key in self.feature_map.keys():
            cur_feature = self.feature_map[key]
            if cur_feature["active"]:
                bbox = cur_feature["bboxes"][-1]
                draw_bbox_text(image, bbox, "%d" % (key))
    # end draw_bboxes


def distance_face(feature_map, feature, threshold, infinity_value):
    """
    Tính khoảng cách giữa vector đặc trưng trung bình của mỗi track (trong feature_map)
    và vector đặc trưng của detection mới, dùng khoảng cách cosine.
    Nếu khoảng cách vượt quá threshold, trả về infinity_value.
    """
    distances = {}
    for key in feature_map.keys():
        key_features = feature_map[key]["features"]
        # Tính vector trung bình của các đặc trưng trong track
        key_fvecs = np.array(key_features).sum(axis=0) / len(key_features)
        # Tính khoảng cách cosine
        distance = scipy.spatial.distance.cosine(key_fvecs, feature)
        if distance <= threshold:
            distances[key] = distance
        else:
            distances[key] = infinity_value
    return distances


def box_text(image, text, point, font_face, font_scale, text_color, thickness, box_color, margin=5):
    size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    (x, y) = point
    text_width = size[0]
    text_height = size[1]
    cv2.rectangle(image, (x - margin, y - text_height - baseline - margin),
                  (x + text_width + margin, y + margin), box_color, cv2.FILLED)
    cv2.putText(image, text, (x, y - baseline), font_face, font_scale, text_color, thickness)


def draw_bbox_text(image, bbox, text):
    (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    box_text(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2, (0, 255, 0), 5)
