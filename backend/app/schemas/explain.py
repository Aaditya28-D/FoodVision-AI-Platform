from pydantic import BaseModel


class ExplainResponse(BaseModel):
    model_name: str
    predicted_class: str
    confidence: float
    heatmap_path: str