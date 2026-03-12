from pydantic import BaseModel
from typing import List


class RetrievalItem(BaseModel):
    rank: int
    class_name: str
    image_path: str
    similarity: float
    image_url: str


class RetrievalResponse(BaseModel):
    top_k: int
    predicted_class: str | None
    exact_match_found: bool
    same_class_results: List[RetrievalItem]
    other_results: List[RetrievalItem]