from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


class ImageEmbedder:
    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def embed_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        features = features.flatten(start_dim=1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu()

    def embed_path(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path)
        return self.embed_pil(image)