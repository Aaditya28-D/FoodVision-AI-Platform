from fastapi import APIRouter

from app.core.config import settings
from ml.inference.class_names import load_class_names
from ml.inference.model_loader import ModelLoader

router = APIRouter(prefix="/models", tags=["Models"])

class_names = load_class_names(settings.CLASS_NAMES_PATH)
model_loader = ModelLoader(num_classes=len(class_names))


@router.get("")
def list_models():
    return {"available_models": model_loader.get_available_models()}