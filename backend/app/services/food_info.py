import json
from pathlib import Path
from typing import Any

from app.schemas.food_info import FoodProfile


class FoodInfoService:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self._profiles = self._load_profiles()

    def _load_profiles(self) -> dict[str, Any]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Food profile data not found: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_profile(self, class_name: str) -> FoodProfile | None:
        raw_profile = self._profiles.get(class_name)
        if raw_profile is None:
            return None
        return FoodProfile(**raw_profile)