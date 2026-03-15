import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbeddingExtractor:

    def __init__(self):

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(320,320))


    def get_embedding(self, img):

        faces = self.app.get(img)

        if len(faces) == 0:
            return None

        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
        )

        embedding = largest_face.embedding

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding


    def get_embedding_from_path(self, img_path):

        img = cv2.imread(img_path)

        if img is None:
            print("Không đọc được ảnh")
            return None

        return self.get_embedding(img)