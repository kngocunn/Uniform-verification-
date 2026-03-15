import cv2
import numpy as np
from ultralytics import YOLO


class PersonCropper:

    def __init__(self, model_path="yolov8n-pose.pt", conf_thres=0.5):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def crop_upper_body(self, image):

        h, w, _ = image.shape
        results = self.model(image)

        for r in results:

            if r.keypoints is None or r.boxes is None:
                return None

            keypoints = r.keypoints.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            best_idx = -1
            max_area = 0

            for i, box in enumerate(boxes):

                if scores[i] < self.conf_thres:
                    continue

                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    best_idx = i

            if best_idx == -1:
                return None

            kpts = keypoints[best_idx]

            nose = kpts[0]
            left_shoulder = kpts[5]
            right_shoulder = kpts[6]

            if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
                return None

            shoulder_center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
            shoulder_center_y = int((left_shoulder[1] + right_shoulder[1]) / 2)

            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

            top = int(nose[1] - 0.6 * shoulder_width)
            bottom = int(shoulder_center_y + 0.8 * shoulder_width)

            left = int(shoulder_center_x - 1.0 * shoulder_width)
            right = int(shoulder_center_x + 1.0 * shoulder_width)

            top = max(0, top)
            bottom = min(h, bottom)
            left = max(0, left)
            right = min(w, right)

            crop = image[top:bottom, left:right]

            return crop

        return None