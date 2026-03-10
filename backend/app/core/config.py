from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "FoodVision AI Platform"
    PROJECT_DESCRIPTION: str = (
        "Backend API for food classification, explainability, "
        "retrieval, and nutrition intelligence."
    )
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
    PROJECT_ROOT: Path = BACKEND_DIR.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    METADATA_DIR: Path = DATA_DIR / "metadata"
    CLASS_NAMES_PATH: Path = METADATA_DIR / "classes.txt"

    MODEL_DIR: Path = BACKEND_DIR / "models"
    MOBILENET_WEIGHTS_PATH: Path = MODEL_DIR / "mobilenet_v3_large_best.pth"
    EFFICIENTNET_B0_WEIGHTS_PATH: Path = MODEL_DIR / "efficientnet_b0_best.pth"
    GOOGLENET_WEIGHTS_PATH: Path = MODEL_DIR / "googlenet_best.pth"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()