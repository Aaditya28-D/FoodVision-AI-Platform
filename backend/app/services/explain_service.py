from app.core.config import settings
from app.schemas.explain import (
    BattleModeResponse,
    BattleModeResult,
    CompareExplainResponse,
    ExplainResponse,
)
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor


class ExplainService:
    def __init__(self) -> None:
        self.explainer = GradCAMExplainer()
        self.predictor = FoodPredictor()

    def explain_single(
        self,
        image,
        model_name: ModelName,
    ) -> ExplainResponse:
        result = self.explainer.explain(
            image=image,
            output_dir=settings.GRADCAM_DIR,
            model_name=model_name,
        )

        return ExplainResponse(
            model_name=result["model_name"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            heatmap_path=result["heatmap_path"],
            heatmap_url=f"/artifacts/gradcam/{result['heatmap_filename']}",
        )

    def explain_compare(
        self,
        image,
    ) -> CompareExplainResponse:
        models = [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ]

        results = [self.explain_single(image=image, model_name=model_name) for model_name in models]
        return CompareExplainResponse(results=results)

    def battle_mode(
        self,
        image,
        top_k: int = 5,
    ) -> BattleModeResponse:
        comparison_response = self.predictor.compare_models(
            image=image,
            top_k=top_k,
        )

        explanation_map = {}
        for model_name in [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ]:
            explanation = self.explain_single(image=image, model_name=model_name)
            explanation_map[model_name.value] = explanation

        merged_results = []
        for comparison_item in comparison_response.results:
            merged_results.append(
                BattleModeResult(
                    comparison=comparison_item,
                    explanation=explanation_map[comparison_item.model_name],
                )
            )

        return BattleModeResponse(
            top_k=comparison_response.top_k,
            results=merged_results,
            summary=comparison_response.summary,
        )
