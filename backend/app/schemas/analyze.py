from pydantic import BaseModel

from app.schemas.explain import BattleModeResponse
from app.schemas.food_info import FoodProfile


class AnalyzeResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    confidence_label: str
    short_summary: str
    food_profile: FoodProfile | None
    battle: BattleModeResponse