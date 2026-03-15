from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import PredictionService
from app.utils.image import validate_and_read_image
from ml.inference.strategy_registry import get_default_strategy, get_strategy_keys

router = APIRouter(prefix="/predict", tags=["Prediction"])

prediction_service = PredictionService()


@router.post("", response_model=PredictionResponse)
async def predict_food(
    image: UploadFile = File(...),
    strategy: str = Query(default=get_default_strategy()),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)
        return prediction_service.predict(
            image=pil_image,
            strategy=strategy,
            top_k=top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid strategy. Use one of: "
                + ", ".join(get_strategy_keys())
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
