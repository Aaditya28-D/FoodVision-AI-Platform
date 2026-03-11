from fastapi import APIRouter
from app.api.routes.comparison import router as comparison_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.prediction import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(prediction_router)
api_router.include_router(comparison_router)