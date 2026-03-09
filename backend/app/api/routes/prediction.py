from fastapi import APIRouter, File, UploadFile
from app.schemas.prediction import PredictionItem, PredictionResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
async def predict_food(image: UploadFile = File(...)):
    _ = image  # placeholder until real inference is connected

    return PredictionResponse(
        model_name="mobilenet_v3_large",
        top_k=5,
        predictions=[
            PredictionItem(class_name="pizza", confidence=0.82),
            PredictionItem(class_name="burger", confidence=0.09),
            PredictionItem(class_name="sushi", confidence=0.04),
            PredictionItem(class_name="ramen", confidence=0.03),
            PredictionItem(class_name="steak", confidence=0.02),
        ],
        inference_time_ms=11.84,
    )