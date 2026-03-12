from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.retrieval import RetrievalItem, RetrievalResponse
from app.utils.image import validate_and_read_image
from ml.inference.predictor import FoodPredictor
from ml.retrieval.retriever import SimilarDishRetriever

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

retriever = SimilarDishRetriever(
    index_path=Path("..") / "data" / "embeddings" / "food101_resnet50_index.npz",
    dataset_root=Path("..") / "data" / "food-101" / "images",
    device="auto",
)

predictor = FoodPredictor()


def to_item(raw: dict) -> RetrievalItem:
    return RetrievalItem(
        rank=raw["rank"],
        class_name=raw["class_name"],
        image_path=raw["image_path"],
        similarity=round(raw["similarity"], 6),
        image_url=f"/dataset-images/{raw['image_path']}",
    )


@router.post("/similar", response_model=RetrievalResponse)
async def retrieve_similar_dishes(
    image: UploadFile = File(...),
    top_k: int = Query(default=6, ge=1, le=20),
):
    try:
        pil_image = await validate_and_read_image(image)

        prediction = predictor.predict(image=pil_image, top_k=1)
        predicted_class = prediction.predictions[0].class_name if prediction.predictions else None

        results = retriever.retrieve(
            image=pil_image,
            top_k=top_k,
            predicted_class=predicted_class,
        )

        return RetrievalResponse(
            top_k=top_k,
            predicted_class=results["predicted_class"],
            exact_match_found=results["exact_match_found"],
            same_class_results=[to_item(item) for item in results["same_class_results"]],
            other_results=[to_item(item) for item in results["other_results"]],
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc