from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.retrieval import RetrievalResponse
from app.services.retrieval_service import RetrievalService
from app.utils.image import validate_and_read_image
from ml.inference.strategy_registry import get_default_strategy, get_strategy_keys

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])

retrieval_service = RetrievalService()


@router.post("/similar", response_model=RetrievalResponse)
async def retrieve_similar_dishes(
    image: UploadFile = File(...),
    top_k: int = Query(default=6, ge=1, le=20),
    strategy: str = Query(default=get_default_strategy()),
):
    try:
        pil_image = await validate_and_read_image(image)
        return retrieval_service.retrieve_similar(
            image=pil_image,
            top_k=top_k,
            strategy=strategy,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid strategy. Use one of: " + ", ".join(get_strategy_keys()),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
