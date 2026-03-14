import torch
from torchvision import transforms
from PIL import Image
import cv2
import os


class LogoClassifier:

    def __init__(self, model_path="logo_classifier.pt", device=None):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(base_dir, "models", model_path)

        # load full model
        self.model = torch.load(model_full_path, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()

        # transform giống lúc train
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        # classes (ImageFolder sẽ sort theo alphabet)
        self.classes = [
            "no_logo",
            "logo"
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