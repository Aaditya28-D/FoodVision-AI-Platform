from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.explain import BattleModeResponse
from app.services.explain_service import ExplainService
from app.utils.image import validate_and_read_image

router = APIRouter(prefix="/battle", tags=["Battle Mode"])

explain_service = ExplainService()


@router.post("", response_model=BattleModeResponse)
async def battle_mode(
    image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=10),
):
    try:
        pil_image = await validate_and_read_image(image)
        return explain_service.battle_mode(
            image=pil_image,
            top_k=top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Battle mode failed: {exc}") from exc
