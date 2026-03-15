import cv2
import numpy as np

from cropper.person_cropper import PersonCropper
from cropper.head_cropper import HeadCropper
from cropper.logo_cropper import LogoCropper
from cropper.boder_cropper import crop_border

from helmet_classifier import HelmetClassifier
from logo_classifier import LogoClassifier
from fake_image_detector import FakeImageDetector

from get_embed.get_face_embedding import FaceEmbeddingExtractor
from get_embed.get_embedding_mysql import EmbeddingDatabase


class VisionPipeline:

    def __init__(self):

        print("Loading models...")

        self.person_cropper = PersonCropper()
        self.head_cropper = HeadCropper()
        self.logo_cropper = LogoCropper()

        self.helmet_classifier = HelmetClassifier()
        self.logo_classifier = LogoClassifier()

        self.fake_detector = FakeImageDetector()

        # ===== FACE MODULE =====
        self.face_embedding = FaceEmbeddingExtractor()
        self.embedding_db = EmbeddingDatabase()

        print("All models loaded successfully!")

    # ================= COSINE SIMILARITY =================
    def cosine_similarity(self, a, b):

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return np.dot(a, b) / (norm_a * norm_b)

    # ================= MAIN PIPELINE =================
    def run(self, image_path, drv_id):

        # ===== VALIDATE drv_id =====
        if drv_id is None:
            return {"error": "drv_id is required"}

        drv_id = str(drv_id).strip()

        if drv_id == "":
            return {"error": "drv_id cannot be empty"}

        image = cv2.imread(image_path)

        if image is None:
            return {"error": "Cannot read image"}

        result = {}

        # ===== PERSON DETECTION =====
        person = self.person_cropper.crop_upper_body(image)

        if person is None:
            result["person_detected"] = False
            return result

        result["person_detected"] = True

        # ===== CROP MODULE =====
        head = self.head_cropper.crop_head(person)
        logo = self.logo_cropper.crop_logo(person)
        border = crop_border(image)

        # ===== HELMET =====
        result["helmet"] = (
            self.helmet_classifier.predict_image(head)
            if head is not None else "head_not_detected"
        )

        # ===== LOGO =====
        result["logo"] = (
            self.logo_classifier.predict_image(logo)
            if logo is not None else "not_detected"
        )

        # ===== FAKE SCREEN =====
        result["fake_screen"] = self.fake_detector.predict(border)

        # ===== FACE EMBEDDING =====
        face_embedding = self.face_embedding.get_embedding(image)

        if face_embedding is None:
            result["face_match"] = False
            result["face_similarity"] = 0
            return result

        # ===== LOAD EMBEDDING FROM DB =====
        db_embedding = self.embedding_db.get_embedding_by_id(drv_id)

        if db_embedding is None:
            result["face_match"] = False
            result["face_similarity"] = 0
            return result

        # ===== COSINE SIMILARITY =====
        similarity = float(self.cosine_similarity(face_embedding, db_embedding))

        result["face_similarity"] = similarity
        result["face_match"] = similarity >= 0.5

        return result
