from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.analyze import AnalyzeResponse
from app.services.analysis_service import AnalysisService
from app.utils.image import validate_and_read_image

router = APIRouter(prefix="/analyze", tags=["Analyze"])

analysis_service = AnalysisService()


@router.post("", response_model=AnalyzeResponse)
async def analyze_food_image(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)
        return analysis_service.analyze(
            image=pil_image,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc
