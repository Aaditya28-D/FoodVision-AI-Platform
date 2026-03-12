from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.retrieval import RetrievalItem, RetrievalResponse
from app.utils.image import validate_and_read_image
from ml.retrieval.retriever import SimilarDishRetriever

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

retriever = SimilarDishRetriever(
    index_path=Path("..") / "data" / "embeddings" / "food101_resnet50_index.npz",
    dataset_root=Path("..") / "data" / "food-101" / "images",
    device="auto",
)


@router.post("/similar", response_model=RetrievalResponse)
async def retrieve_similar_dishes(
    image: UploadFile = File(...),
    top_k: int = Query(default=6, ge=1, le=20),
):
    try:
        pil_image = await validate_and_read_image(image)
        results = retriever.retrieve(image=pil_image, top_k=top_k)

        response_items = []
        for item in results:
            response_items.append(
                RetrievalItem(
                    rank=item["rank"],
                    class_name=item["class_name"],
                    image_path=item["image_path"],
                    similarity=round(item["similarity"], 6),
                    image_url=f"/dataset-images/{item['image_path']}",
                )
            )

        return RetrievalResponse(top_k=top_k, results=response_items)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc