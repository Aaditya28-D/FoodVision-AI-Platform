from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.bulk_classification import BulkClassificationSummary
from app.services.bulk_classification_service import BulkClassificationService
from app.utils.image import validate_and_read_image

router = APIRouter(prefix="/bulk-classify", tags=["Bulk Classification"])

bulk_service = BulkClassificationService()


@router.post("", response_model=BulkClassificationSummary)
async def bulk_classify_images(
    files: list[UploadFile] = File(...),
    strategy: str = Form(default="ensemble"),
    confidence_threshold: float = Form(default=0.6),
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        prepared_files: list[tuple[str, bytes]] = []

        for file in files:
            contents = await file.read()
            await file.seek(0)

            # quick validation using existing image logic
            await validate_and_read_image(file)

            prepared_files.append((file.filename or "image.jpg", contents))

        result = bulk_service.classify_files(
            files=prepared_files,
            strategy=strategy,
            confidence_threshold=confidence_threshold,
        )
        return BulkClassificationSummary(**result)

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bulk classification failed: {exc}") from exc
