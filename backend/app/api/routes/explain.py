from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.explain import CompareExplainResponse, ExplainResponse
from app.utils.image import validate_and_read_image
from ml.explainability.gradcam import GradCAMExplainer
from ml.inference.model_registry import ModelName

router = APIRouter(prefix="/explain", tags=["Explainability"])

explainer = GradCAMExplainer()


@router.post("", response_model=ExplainResponse)
async def explain_food_image(
    image: UploadFile = File(...),
    model_name: ModelName = Query(default=ModelName.EFFICIENTNET_B0),
):
    try:
        pil_image = await validate_and_read_image(image)
        output_dir = Path("artifacts") / "gradcam"
        result = explainer.explain(
            image=pil_image,
            output_dir=output_dir,
            model_name=model_name,
        )

        return ExplainResponse(
            model_name=result["model_name"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            heatmap_path=result["heatmap_path"],
            heatmap_url=f"/artifacts/gradcam/{result['heatmap_filename']}",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {exc}") from exc


@router.post("/compare", response_model=CompareExplainResponse)
async def compare_explanations(
    image: UploadFile = File(...),
):
    try:
        pil_image = await validate_and_read_image(image)
        output_dir = Path("artifacts") / "gradcam"

        models = [
            ModelName.EFFICIENTNET_B0,
            ModelName.RESNET50,
            ModelName.MOBILENET_V3_LARGE,
        ]

        results = []
        for model_name in models:
            result = explainer.explain(
                image=pil_image,
                output_dir=output_dir,
                model_name=model_name,
            )

            results.append(
                ExplainResponse(
                    model_name=result["model_name"],
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    heatmap_path=result["heatmap_path"],
                    heatmap_url=f"/artifacts/gradcam/{result['heatmap_filename']}",
                )
            )

        return CompareExplainResponse(results=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison explainability failed: {exc}") from exc