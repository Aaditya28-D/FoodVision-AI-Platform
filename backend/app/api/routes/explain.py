from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.explain import CompareExplainResponse, ExplainResponse
from app.services.explain_service import ExplainService
from app.utils.image import validate_and_read_image
from ml.inference.model_registry import ModelName

router = APIRouter(prefix="/explain", tags=["Explainability"])

explain_service = ExplainService()


@router.post("", response_model=ExplainResponse)
async def explain_food_image(
    image: UploadFile = File(...),
    model_name: ModelName = Query(default=ModelName.EFFICIENTNET_B0),
):
    try:
        pil_image = await validate_and_read_image(image)
        return explain_service.explain_single(
            image=pil_image,
            model_name=model_name,
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
        return explain_service.explain_compare(image=pil_image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison explainability failed: {exc}") from exc
