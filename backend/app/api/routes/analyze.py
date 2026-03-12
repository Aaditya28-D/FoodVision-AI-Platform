from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.analyze import AnalyzeResponse
from app.schemas.explain import BattleModeResponse, BattleModeResult, ExplainResponse
from app.services.food_info import FoodInfoService
from app.utils.confidence import confidence_label
from app.utils.image import validate_and_read_image
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/analyze", tags=["Analyze"])

predictor = FoodPredictor()
explainer = GradCAMExplainer()
food_info_service = FoodInfoService(data_dir=Path("..") / "data" / "food_info" / "profiles")


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


@router.post("", response_model=AnalyzeResponse)
async def analyze_food_image(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)

        smart_prediction = predictor.predict_smart(
            image=pil_image,
            top_k=top_k,
        )

        comparison_response = predictor.compare_models(
            image=pil_image,
            top_k=top_k,
        )

        output_dir = Path("artifacts") / "gradcam"
        explanation_map = {}

        for model_name in [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ]:
            result = explainer.explain(
                image=pil_image,
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

        top_prediction = smart_prediction.predictions[0]
        predicted_class = top_prediction.class_name
        confidence = top_prediction.confidence
        confidence_text = confidence_label(confidence)
        food_profile = food_info_service.get_profile(predicted_class)
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

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc