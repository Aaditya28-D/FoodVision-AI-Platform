from pydantic import BaseModel
from typing import List


class ExplainResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    heatmap_path: str
    heatmap_url: str


class CompareExplainResponse(BaseModel):
    results: List[ExplainResponse]