import cv2
import mediapipe as mp


class LogoCropper:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        )

    def crop_logo(self, body_img):

        h, w, _ = body_img.shape

        img_rgb = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        x_ls = int(left_shoulder.x * w)
        y_ls = int(left_shoulder.y * h)

        x_rs = int(right_shoulder.x * w)

        shoulder_width = abs(x_ls - x_rs)

        top = int(y_ls + 0 * shoulder_width)
        bottom = int(y_ls + 0.6 * shoulder_width)

        left = int(x_ls - 0.6 * shoulder_width)
        right = int(x_ls + 0.2 * shoulder_width)

        top = max(0, top)
        bottom = min(h, bottom)
        left = max(0, left)
        right = min(w, right)

        logo = body_img[top:bottom, left:right]

        return logo