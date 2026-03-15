from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.retrieval import RetrievalResponse
from app.services.retrieval_service import RetrievalService
from app.utils.image import validate_and_read_image

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

retrieval_service = RetrievalService()


@router.post("/similar", response_model=RetrievalResponse)
async def retrieve_similar_dishes(
    image: UploadFile = File(...),
    top_k: int = Query(default=6, ge=1, le=20),
):
    try:
        pil_image = await validate_and_read_image(image)
        return retrieval_service.retrieve_similar(
            image=pil_image,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
