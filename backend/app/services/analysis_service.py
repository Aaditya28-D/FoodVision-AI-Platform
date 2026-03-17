from pathlib import Path

from app.schemas.analyze import AnalyzeResponse
from app.schemas.explain import BattleModeResponse, BattleModeResult, ExplainResponse
from app.services.food_info import FoodInfoService
from app.services.prediction_service import PredictionService
from app.utils.confidence import confidence_label
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.strategy_registry import (
    get_default_strategy,
    get_strategy_models,
)


class AnalysisService:
    def __init__(self) -> None:
        self.prediction_service = PredictionService()
        self.explainer = GradCAMExplainer()
        self.food_info_service = FoodInfoService(
            data_dir=Path("..") / "data" / "food_info" / "profiles"
        )

    def build_short_summary(
        self,
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

    def analyze(
        self,
        image,
        top_k: int = 5,
        strategy: str | None = None,
    ) -> AnalyzeResponse:
        prediction = self.prediction_service.predict(
            image=image,
            strategy=strategy,
            top_k=top_k,
        )

        selected_strategy = strategy or get_default_strategy()
        explain_models = get_strategy_models(selected_strategy)

        comparison_response = self.prediction_service.predictor.compare_specific_models(
            image=image,
            model_names=explain_models,
            top_k=top_k,
        )

        output_dir = Path("artifacts") / "gradcam"
        explanation_map = {}

        for model_name in explain_models:
            result = self.explainer.explain(
                image=image,
                output_dir=output_dir,
                model_name=model_name,
            )

            explanation_map[model_name.value] = ExplainResponse(
                model_name=result["model_name"],
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                heatmap_path=result["heatmap_path"],
                heatmap_url=f"/artifacts/gradcam/{result['heatmap_filename']}",
            )

        merged_results = []
        for comparison_item in comparison_response.results:
            merged_results.append(
                BattleModeResult(
                    comparison=comparison_item,
                    explanation=explanation_map[comparison_item.model_name],
                )
            )

        battle_response = BattleModeResponse(
            top_k=comparison_response.top_k,
            results=merged_results,
            summary=comparison_response.summary,
        )

        top_prediction = prediction.predictions[0]
        predicted_class = top_prediction.class_name
        confidence = top_prediction.confidence
        confidence_text = confidence_label(confidence)
        food_profile = self.food_info_service.get_profile(predicted_class)
        summary = self.build_short_summary(predicted_class, confidence_text, food_profile)

        return AnalyzeResponse(
            model_name=prediction.model_name,
            predicted_class=predicted_class,
            confidence=confidence,
            confidence_label=confidence_text,
            short_summary=summary,
            food_profile=food_profile,
            battle=battle_response,
        )
