from fastapi import APIRouter, File, UploadFile, Query

from app.schemas.prediction import PredictionResponse
from app.utils.image import validate_and_read_image
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/predict", tags=["Prediction"])

predictor = FoodPredictor()


@router.post("", response_model=PredictionResponse)
async def predict_food(
    image: UploadFile = File(...),
    model_name: ModelName = Query(default=ModelName.MOBILENET_V3_LARGE),
    top_k: int = Query(default=5, ge=1, le=10),
):
    pil_image = await validate_and_read_image(image)
    response = predictor.predict(
        image=pil_image,
        model_name=model_name,
        top_k=top_k,
    )
    return response