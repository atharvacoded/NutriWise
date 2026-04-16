"""
backend/schemas.py
──────────────────
Pydantic v2 models for API request / response validation.
"""

from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, model_validator  # pyright: ignore[reportMissingImports]


# ─── Request ──────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    # Demographics
    age:           int            = Field(..., ge=10, le=100)
    sex:           Literal["male", "female"]
    height_cm:     float          = Field(..., ge=100, le=250)
    weight_kg:     float          = Field(..., ge=20, le=300)
    body_fat_pct:  Optional[float] = Field(None, ge=3, le=65)

    # Lifestyle
    activity_level: Literal["sedentary", "light", "moderate", "very", "extreme"]
    sleep_h:        float  = Field(7.0, ge=3, le=12)
    meals_per_day:  int    = Field(3, ge=2, le=6)

    # Goal
    goal:           Literal["loss", "muscle", "gain"]
    target_weight_kg: Optional[float] = None
    timeline_weeks:   Optional[int]   = Field(None, ge=2, le=104)

    # Diet
    diet_type:    Literal["none", "vegetarian", "vegan", "eggetarian", "glutenFree", "dairyFree"] = "none"
    cuisine:      Literal["india", "western", "asian", "mediterranean", "any"] = "any"
    allergies:    Optional[str] = None
    health_notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_target_weight(self) -> UserProfile:
        if self.target_weight_kg is not None:
            if self.goal == "loss" and self.target_weight_kg >= self.weight_kg:
                raise ValueError("Target weight must be less than current weight for loss goal")
            if self.goal in ("gain", "muscle") and self.target_weight_kg <= self.weight_kg:
                raise ValueError("Target weight must be greater than current weight for gain/muscle goal")
        return self


class UserAuth(BaseModel):
    email: str
    password: str



# ─── Response ─────────────────────────────────────────────────────────────────

class MacroBreakdown(BaseModel):
    protein_g:    float
    carbs_g:      float
    fat_g:        float
    protein_pct:  float
    carbs_pct:    float
    fat_pct:      float


class FoodItem(BaseModel):
    name:         str
    quantity_g:   float
    calories:     float
    protein_g:    float
    carbs_g:      float
    fat_g:        float
    prep_note:    Optional[str] = None
    score:        Optional[float] = None   # 0–100 suitability


class Meal(BaseModel):
    meal_name:    str
    time_window:  str
    total_calories: float
    foods:        List[FoodItem]


class WeeklyProjection(BaseModel):
    week:           int
    calorie_target: float
    protein_g:      float
    carbs_g:        float
    fat_g:          float
    projected_weight_kg: Optional[float] = None


class NutritionPlan(BaseModel):
    # Core metrics
    bmr:            float
    tdee:           float
    calorie_target: float
    bmi:            float
    bmi_category:   str
    macros:         MacroBreakdown
    meals:          List[Meal]

    # Projections
    weekly_projections: List[WeeklyProjection]

    # Guidance
    hydration_ml:   int
    tips:           List[str]
    supplements:    List[str]
    warnings:       List[str]

    # Meta
    goal_code:      Literal["loss", "muscle", "gain"]
    goal_label:     str
    model_version:  str = "1.0"


class FoodSearchResult(BaseModel):
    fdc_id:       Optional[int] = None
    name:         str
    category:     str
    calories:     float
    protein_g:    float
    carbs_g:      float
    fat_g:        float
    fiber_g:      float
    score:        Optional[float] = None
    source:       str = "USDA"
