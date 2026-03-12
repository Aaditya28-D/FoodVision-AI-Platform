from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router

app = FastAPI(
    title="FoodVision AI Platform",
    version="0.1.0",
    description="Backend API for food classification, explainability, retrieval, and nutrition intelligence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")
app.mount(
    "/dataset-images",
    StaticFiles(directory=str(Path("..") / "data" / "food-101" / "images")),
    name="dataset-images",
)


@app.get("/")
def root():
    return {"message": "FoodVision AI Platform backend is running"}