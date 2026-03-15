from fastapi import APIRouter

from app.schemas.strategy import StrategyItem, StrategyListResponse
from ml.inference.strategy_registry import (
    get_default_strategy,
    get_strategy_definitions,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=StrategyListResponse)
def list_models():
    strategies = [
        StrategyItem(
            key=item.key,
            label=item.label,
            description=item.description,
            category=item.category,
            is_default=item.is_default,
        )
        for item in get_strategy_definitions()
    ]

    return StrategyListResponse(
        default_strategy=get_default_strategy(),
        available_strategies=strategies,
    )
