import json
from pathlib import Path

from app.schemas.food_info import FoodProfile


class FoodInfoService:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._profiles = self._load_profiles()

    def _load_profiles(self) -> dict[str, dict]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Food profile directory not found: {self.data_dir}")

        profiles: dict[str, dict] = {}

        for file_path in sorted(self.data_dir.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as file:
                profiles[file_path.stem] = json.load(file)

        return profiles

    def get_profile(self, class_name: str) -> FoodProfile | None:
        raw_profile = self._profiles.get(class_name)
        if raw_profile is None:
            return None
        return FoodProfile(**raw_profile)