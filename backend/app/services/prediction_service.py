from app.schemas.prediction import PredictionResponse
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor


class PredictionService:
    def __init__(self) -> None:
        self.predictor = FoodPredictor()

    def predict(
        self,
        image,
        model_name: str = "smart",
        top_k: int = 5,
    ) -> PredictionResponse:
        if model_name == "smart":
            return self.predictor.predict_smart(image=image, top_k=top_k)

        if model_name == "ensemble":
            return self.predictor.predict_ensemble(image=image, top_k=top_k)

        return self.predictor.predict(
            image=image,
            model_name=ModelName(model_name),
            top_k=top_k,
        )
