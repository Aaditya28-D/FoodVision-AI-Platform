from time import perf_counter
from typing import List

import torch
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import PredictionItem, PredictionResponse
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader
from ml.inference.model_registry import ModelName
from ml.inference.transforms import get_inference_transforms


class FoodPredictor:
    def __init__(self) -> None:
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)
        self.model_loader = ModelLoader(num_classes=len(self.class_names))
        self.transforms = get_inference_transforms(image_size=224)

    def predict(
        self,
        image: Image.Image,
        model_name: ModelName = ModelName.MOBILENET_V3_LARGE,
        top_k: int = 5,
    ) -> PredictionResponse:
        start_time = perf_counter()

        loaded_model = self.model_loader.load_model(model_name)

        image_tensor = self.transforms(image).unsqueeze(0).to(loaded_model.device)

        with torch.no_grad():
            outputs = loaded_model.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        predictions = [
            PredictionItem(
                class_name=self.class_names[idx],
                confidence=round(float(prob), 6),
            )
            for prob, idx in zip(top_probs, top_indices)
        ]

        inference_time_ms = (perf_counter() - start_time) * 1000

        return PredictionResponse(
            model_name=loaded_model.model_name,
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
            device=loaded_model.device,
        )