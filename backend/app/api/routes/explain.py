from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.explain import ExplainResponse
from app.utils.image import validate_and_read_image
from ml.explainability.gradcam import GradCAMExplainer

router = APIRouter(prefix="/explain", tags=["Explainability"])

explainer = GradCAMExplainer()


@router.post("", response_model=ExplainResponse)
async def explain_food_image(
    image: UploadFile = File(...),
):
    try:
        pil_image = await validate_and_read_image(image)
        output_dir = Path("artifacts") / "gradcam"
        result = explainer.explain(
            image=pil_image,
            output_dir=output_dir,
        )
        return ExplainResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {exc}") from exc