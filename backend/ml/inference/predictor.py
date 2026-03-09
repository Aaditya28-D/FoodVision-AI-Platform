from time import perf_counter
from typing import List

from PIL import Image

from app.core.config import settings
from app.schemas.prediction import PredictionItem, PredictionResponse
from ml.inference.class_names import load_class_names


class FoodPredictor:
    def __init__(self, model_name: str = "mobilenet_v3_large") -> None:
        self.model_name = model_name
        self.class_names: List[str] = load_class_names(settings.CLASS_NAMES_PATH)

    def predict(self, image: Image.Image, top_k: int = 5) -> PredictionResponse:
        _ = image

        start_time = perf_counter()

        selected_classes = self.class_names[:top_k]
        dummy_confidences = [0.82, 0.09, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002]

        predictions = [
            PredictionItem(class_name=class_name, confidence=dummy_confidences[idx])
            for idx, class_name in enumerate(selected_classes)
        ]

        inference_time_ms = (perf_counter() - start_time) * 1000

        return PredictionResponse(
            model_name=self.model_name,
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
        )