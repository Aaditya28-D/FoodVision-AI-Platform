from app.core.config import settings
from app.schemas.analyze import AnalyzeResponse
from app.services.explain_service import ExplainService
from app.services.food_info import FoodInfoService
from app.utils.confidence import confidence_label
from ml.inference.predictor import FoodPredictor


def build_short_summary(
    predicted_class: str,
    confidence_text: str,
    food_profile,
) -> str:
    food_name = predicted_class.replace("_", " ").title()

    if food_profile is None:
        return (
            f"The model is {confidence_text.lower()} that this image shows {food_name}. "
            f"Detailed food profile information is not available yet for this class."
        )

    return (
        f"The model is {confidence_text.lower()} that this image shows {food_profile.food_name}. "
        f"{food_profile.food_name} is a {food_profile.category.lower()} dish from {food_profile.cuisine} cuisine. "
        f"{food_profile.description} "
        f"This food can appear in different variations, so the exact style may vary from image to image."
    )


class AnalysisService:
    def __init__(self) -> None:
        self.predictor = FoodPredictor()
        self.explain_service = ExplainService()
        self.food_info_service = FoodInfoService(data_dir=settings.FOOD_INFO_DIR)

    def analyze(
        self,
        image,
        top_k: int = 5,
    ) -> AnalyzeResponse:
        smart_prediction = self.predictor.predict_smart(
            image=image,
            top_k=top_k,
        )

        battle_response = self.explain_service.battle_mode(
            image=image,
            top_k=top_k,
        )

        top_prediction = smart_prediction.predictions[0]
        predicted_class = top_prediction.class_name
        confidence = top_prediction.confidence
        confidence_text = confidence_label(confidence)
        food_profile = self.food_info_service.get_profile(predicted_class)
        summary = build_short_summary(predicted_class, confidence_text, food_profile)

        return AnalyzeResponse(
            model_name=smart_prediction.model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            confidence_label=confidence_text,
            short_summary=summary,
            food_profile=food_profile,
            battle=battle_response,
        )
