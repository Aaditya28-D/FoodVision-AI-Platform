from fastapi import FastAPI

app = FastAPI(
    title="FoodVision AI Platform",
    description="Backend API for food classification, explainability, retrieval, and nutrition intelligence.",
    version="0.1.0"
)


@app.get("/")
def root():
    return {"message": "FoodVision AI Platform backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}