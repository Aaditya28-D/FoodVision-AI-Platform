from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.prediction import PredictionResponse
from app.utils.image import validate_and_read_image
from ml.inference.model_registry import ModelName
from ml.inference.predictor import FoodPredictor

router = APIRouter(prefix="/predict", tags=["Prediction"])

predictor = FoodPredictor()


@router.post("", response_model=PredictionResponse)
async def predict_food(
    image: UploadFile = File(...),
    model_name: str = Query(default="ensemble"),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)

        if model_name == "ensemble":
            response = predictor.predict_ensemble(
                image=pil_image,
                top_k=top_k,
            )
        else:
            response = predictor.predict(
                image=pil_image,
                model_name=ModelName(model_name),
                top_k=top_k,
            )

        return response
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model_name. Use 'ensemble' or one of: "
                "mobilenet_v3_large, efficientnet_b0, resnet50, vit_b_16, googlenet."
            ),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc