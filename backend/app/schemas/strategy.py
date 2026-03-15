from pydantic import BaseModel


class StrategyItem(BaseModel):
    key: str
    label: str
    description: str
    category: str
    is_default: bool


class StrategyListResponse(BaseModel):
    default_strategy: str
    available_strategies: list[StrategyItem]
