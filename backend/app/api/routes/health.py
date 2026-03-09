from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {"message": "FoodVision AI Platform backend is running"}


@router.get("/health")
def health_check():
    return {"status": "ok"}