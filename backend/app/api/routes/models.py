from fastapi import APIRouter

from ml.inference.strategy_registry import (
    get_default_strategy,
    get_strategy_metadata,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("")
def list_models():
    return {
        "default_strategy": get_default_strategy(),
        "available_strategies": get_strategy_metadata(),
    }
