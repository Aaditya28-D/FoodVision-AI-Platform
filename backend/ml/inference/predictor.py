from time import perf_counter
from PIL import Image

from app.schemas.prediction import PredictionItem, PredictionResponse


class FoodPredictor:
    def __init__(self, model_name: str = "mobilenet_v3_large") -> None:
        self.model_name = model_name

    def predict(self, image: Image.Image, top_k: int = 5) -> PredictionResponse:
        _ = image

        start_time = perf_counter()

        predictions = [
            PredictionItem(class_name="pizza", confidence=0.82),
            PredictionItem(class_name="burger", confidence=0.09),
            PredictionItem(class_name="sushi", confidence=0.04),
            PredictionItem(class_name="ramen", confidence=0.03),
            PredictionItem(class_name="steak", confidence=0.02),
        ][:top_k]

        inference_time_ms = (perf_counter() - start_time) * 1000

        return PredictionResponse(
            model_name=self.model_name,
            top_k=top_k,
            predictions=predictions,
            inference_time_ms=round(inference_time_ms, 3),
        )