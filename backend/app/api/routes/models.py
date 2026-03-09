from fastapi import APIRouter

from ml.inference.model_loader import ModelLoader

router = APIRouter(prefix="/models", tags=["Models"])

model_loader = ModelLoader()


@router.get("")
def list_models():
    return {"available_models": model_loader.get_available_models()}