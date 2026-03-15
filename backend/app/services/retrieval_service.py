from app.core.config import settings
from app.schemas.retrieval import RetrievalItem, RetrievalResponse
from ml.inference.predictor import FoodPredictor
from ml.retrieval.retriever import SimilarDishRetriever


class RetrievalService:
    def __init__(self) -> None:
        self.retriever = SimilarDishRetriever(
            index_path=settings.RETRIEVAL_INDEX_PATH,
            dataset_root=settings.FOOD101_IMAGES_DIR,
            device="auto",
        )
        self.predictor = FoodPredictor()

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
    ) -> RetrievalResponse:
        smart_prediction = self.predictor.predict_smart(image=image, top_k=1)
        predicted_class = (
            smart_prediction.predictions[0].class_name
            if smart_prediction.predictions
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
