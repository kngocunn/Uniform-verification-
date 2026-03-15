import os
import torch
import cv2
from torchvision import transforms
from PIL import Image


class FakeImageDetector:

    def __init__(self, model_path=None, device=None):

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        base_dir = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            model_path = os.path.join(base_dir, "models", "fake_screen_detector_resnet50.pt")

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = checkpoint["model"]
        self.classes = checkpoint["classes"]

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])


    def predict(self, image):

        # cv2 BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(img)

            probs = torch.softmax(outputs, dim=1)

            conf, pred = torch.max(probs, 1)

        label = self.classes[pred.item()]

        return label
