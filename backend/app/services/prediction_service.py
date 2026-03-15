from app.schemas.prediction import PredictionResponse
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor
from ml.inference.strategy_registry import get_default_strategy


class PredictionService:
    def __init__(self) -> None:
        self.predictor = FoodPredictor()

    def predict(
        self,
        image,
        strategy: str | None = None,
        top_k: int = 5,
    ) -> PredictionResponse:
        selected_strategy = strategy or get_default_strategy()

        if selected_strategy == "smart":
            return self.predictor.predict_smart(image=image, top_k=top_k)

        if selected_strategy == "ensemble":
            return self.predictor.predict_ensemble(image=image, top_k=top_k)

        return self.predictor.predict(
            image=image,
            model_name=ModelName(selected_strategy),
            top_k=top_k,
        )
