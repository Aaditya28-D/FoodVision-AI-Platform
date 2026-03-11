from pydantic import BaseModel
from typing import List


class NutritionInfo(BaseModel):
    calories: str
    protein: str
    carbohydrates: str
    fat: str


class HealthAdvice(BaseModel):
    benefits: str
    risks: str
    healthy_tip: str


class DietaryNotes(BaseModel):
    vegetarian_possible: bool
    non_vegetarian_possible: bool
    vegan_possible: bool
    gluten_free_possible: bool
    note: str


class ServingInfo(BaseModel):
    typical_serving: str
    best_served_with: List[str]


class FoodProfile(BaseModel):
    food_name: str
    category: str
    cuisine: str
    description: str
    core_ingredients: List[str]
    optional_ingredients: List[str]
    preparation_steps: List[str]
    nutrition: NutritionInfo
    health_advice: HealthAdvice
    dietary_notes: DietaryNotes
    allergens: List[str]
    common_variations: List[str]
    serving_info: ServingInfo
    popular_regions: List[str]