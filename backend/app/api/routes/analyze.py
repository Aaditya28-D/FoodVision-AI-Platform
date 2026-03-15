from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.analyze import AnalyzeResponse
from app.services.analysis_service import AnalysisService
from app.utils.image import validate_and_read_image
from ml.inference.strategy_registry import get_default_strategy, get_strategy_keys

router = APIRouter(prefix="/analyze", tags=["Analyze"])

analysis_service = AnalysisService()


@router.post("", response_model=AnalyzeResponse)
async def analyze_food_image(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
    strategy: str = Query(default=get_default_strategy()),
):
    try:
        pil_image = await validate_and_read_image(image)
        return analysis_service.analyze(
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
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc
