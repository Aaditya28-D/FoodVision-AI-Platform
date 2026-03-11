from pydantic import BaseModel
from typing import List

from app.schemas.prediction import BattleSummary, ComparisonResult


class ExplainResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    heatmap_path: str
    heatmap_url: str


class CompareExplainResponse(BaseModel):
    results: List[ExplainResponse]


class BattleModeResult(BaseModel):
    comparison: ComparisonResult
    explanation: ExplainResponse


class BattleModeResponse(BaseModel):
    top_k: int
    results: List[BattleModeResult]
    summary: BattleSummary