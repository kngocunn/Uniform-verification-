import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import cv2
import os


class HelmetClassifier:

    def __init__(self, model_path="helmet_classification.pt", device=None):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(base_dir, "models", model_path)

        model_data = torch.load(model_full_path, map_location=self.device)

        if isinstance(model_data, dict):
            self.model.load_state_dict(model_data)
        else:
            self.model = model_data

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        self.classes = [
            "helmet_1",
            "helmet_2",
            "wrong_helmet"
        ]


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
