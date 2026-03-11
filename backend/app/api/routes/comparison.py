from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.prediction import ComparisonResponse
from app.utils.image import validate_and_read_image
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/compare", tags=["Comparison"])

predictor = FoodPredictor()


@router.post("", response_model=ComparisonResponse)
async def compare_food_models(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)
        response = predictor.compare_models(
            image=pil_image,
            top_k=top_k,
        )
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc