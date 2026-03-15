import cv2
import mediapipe as mp


class HeadCropper:

    def __init__(self):

        mp_face = mp.solutions.face_detection

        self.face_detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def crop_head(self, image):

        h_img, w_img = image.shape[:2]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(img_rgb)

        if not results.detections:
            return None

        detection = results.detections[0]

        bbox = detection.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w_img)
        y1 = int(bbox.ymin * h_img)
        w = int(bbox.width * w_img)
        h = int(bbox.height * h_img)

        x2 = x1 + w
        y2 = y1 + h

        expand_top = int(1.3 * h)
        expand_side = int(0.6 * w)
        expand_bottom = int(0.4 * h)

        new_x1 = max(0, x1 - expand_side)
        new_y1 = max(0, y1 - expand_top)
        new_x2 = min(w_img, x2 + expand_side)
        new_y2 = min(h_img, y2 + expand_bottom)

        head = image[new_y1:new_y2, new_x1:new_x2]

        return head