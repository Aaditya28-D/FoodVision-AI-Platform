from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.explain import (
    BattleModeResponse,
    BattleModeResult,
    ExplainResponse,
)
from app.utils.image import validate_and_read_image
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/battle", tags=["Battle Mode"])

predictor = FoodPredictor()
explainer = GradCAMExplainer()


@router.post("", response_model=BattleModeResponse)
async def battle_mode(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)

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

        return BattleModeResponse(
            top_k=comparison_response.top_k,
            results=merged_results,
            summary=comparison_response.summary,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Battle mode failed: {exc}") from exc