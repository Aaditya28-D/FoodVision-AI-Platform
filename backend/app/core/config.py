from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "FoodVision AI Platform"
    PROJECT_DESCRIPTION: str = (
        "Backend API for food classification, explainability, "
        "retrieval, and nutrition intelligence."
    )
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()