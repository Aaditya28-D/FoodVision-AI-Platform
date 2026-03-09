from fastapi import HTTPException, UploadFile
from PIL import Image
import io

from app.core.constants import ALLOWED_IMAGE_TYPES, MAX_IMAGE_SIZE_BYTES


async def validate_and_read_image(upload_file: UploadFile) -> Image.Image:
    if upload_file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {upload_file.content_type}"
        )

    contents = await upload_file.read()

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image file is too large"
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        ) from exc

    return image