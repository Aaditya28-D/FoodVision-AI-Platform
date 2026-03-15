from pathlib import Path

from app.schemas.retrieval import RetrievalItem, RetrievalResponse
from app.services.prediction_service import PredictionService
from ml.retrieval.retriever import SimilarDishRetriever


class RetrievalService:
    def __init__(self) -> None:
        self.retriever = SimilarDishRetriever(
            index_path=Path("..") / "data" / "embeddings" / "food101_resnet50_index.npz",
            dataset_root=Path("..") / "data" / "food-101" / "images",
            device="auto",
        )
        self.prediction_service = PredictionService()

    def _to_item(self, raw: dict) -> RetrievalItem:
        return RetrievalItem(
            rank=raw["rank"],
            class_name=raw["class_name"],
            image_path=raw["image_path"],
            similarity=round(raw["similarity"], 6),
            image_url=f"/dataset-images/{raw['image_path']}",
        )

    def retrieve_similar(
        self,
        image,
        top_k: int = 6,
        strategy: str | None = None,
    ) -> RetrievalResponse:
        prediction = self.prediction_service.predict(
            image=image,
            strategy=strategy,
            top_k=1,
        )

        predicted_class = (
            prediction.predictions[0].class_name
            if prediction.predictions
            else None
        )

        results = self.retriever.retrieve(
            image=image,
            top_k=top_k,
            predicted_class=predicted_class,
        )

        return RetrievalResponse(
            top_k=top_k,
            predicted_class=results["predicted_class"],
            exact_match_found=results["exact_match_found"],
            same_class_results=[self._to_item(item) for item in results["same_class_results"]],
            other_results=[self._to_item(item) for item in results["other_results"]],
        )
