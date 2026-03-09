from fastapi import APIRouter, File, UploadFile, Query

from app.schemas.prediction import PredictionResponse
from app.utils.image import validate_and_read_image
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/predict", tags=["Prediction"])

predictor = FoodPredictor(model_name="mobilenet_v3_large")


@router.post("", response_model=PredictionResponse)
async def predict_food(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    pil_image = await validate_and_read_image(image)
    response = predictor.predict(pil_image, top_k=top_k)
    return response