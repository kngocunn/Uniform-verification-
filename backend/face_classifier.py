import os
import torch
import cv2
from PIL import Image
from torchvision import transforms


class FaceClassifier:

    def __init__(self, model_path="models/face_classifier.pt", device=None):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(current_dir, model_path)

        self.model = torch.load(model_full_path, map_location=self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

        self.classes = ["false","true"]


    def predict_image(self, image):

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)

        pred = torch.argmax(output,1).item()

        return self.classes[pred]