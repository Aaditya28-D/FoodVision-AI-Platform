from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionItem(BaseModel):
    class_name: str = Field(..., example="pizza")
    confidence: float = Field(..., example=0.91)


class PredictionResponse(BaseModel):
    model_name: str = Field(..., example="mobilenet_v3_large")
    top_k: int = Field(..., example=5)
    predictions: List[PredictionItem]
    inference_time_ms: float = Field(..., example=12.47)
    device: Optional[str] = Field(default="cpu", example="cpu")


class ComparisonResult(BaseModel):
    model_name: str
    predictions: List[PredictionItem]
    inference_time_ms: float
    device: str


class ComparisonResponse(BaseModel):
    top_k: int
    results: List[ComparisonResult]