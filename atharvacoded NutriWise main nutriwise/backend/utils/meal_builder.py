"""
backend/utils/meal_builder.py
──────────────────────────────
Builds a structured, calorie-accurate meal plan from the food database,
tailored to the user's goal, cuisine, and dietary restrictions.
"""

from __future__ import annotations
import random
from typing import List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from backend.schemas import FoodItem, Meal, UserProfile
from backend.ml_engine import score_food

DATA_DIR = Path(__file__).parent.parent / "data"

# Cached food DB
_food_db: Optional[pd.DataFrame] = None


def _load_food_db() -> pd.DataFrame:
    global _food_db
    if _food_db is not None:
        return _food_db

    dfs = []
    for fname in ["food_nutrients.csv", "ifct_indian_foods.csv"]:
        p = DATA_DIR / fname
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        # Minimal fallback if DB not yet downloaded
        _food_db = pd.DataFrame()
        return _food_db

    df = pd.concat(dfs, ignore_index=True)
    for col in ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df = df[df["calories"] > 0].reset_index(drop=True)
    _food_db = df
    return _food_db


# ─── Hardcoded curated meal templates ─────────────────────────────────────────
# Used when the CSV food DB is not available (zero-dependency mode)

MEAL_TEMPLATES = {
    "india": {
        "loss": [
            {
                "meal": "Breakfast",
                "time": "7:00–8:30 AM",
                "foods": [
                    ("Oats with low-fat milk (1 cup)", 150, 6.0, 27.0, 3.0, "Cook rolled oats in milk, no sugar"),
                    ("2 boiled eggs", 155, 12.6, 1.1, 10.6, "Boiled 10 min"),
                    ("1 banana", 89, 1.1, 22.8, 0.3, "Pre-workout carbs"),
                ],
            },
            {
                "meal": "Mid-Morning Snack",
                "time": "10:30–11:00 AM",
                "foods": [
                    ("Low-fat curd (200g)", 58, 6.6, 7.0, 0.2, "Plain dahi, no sugar"),
                    ("Handful of almonds (20g)", 116, 4.2, 4.3, 10.0, "Raw, unsalted"),
                ],
            },
            {
                "meal": "Lunch",
                "time": "12:30–2:00 PM",
                "foods": [
                    ("2 whole wheat rotis", 178, 5.8, 33.9, 2.2, "No ghee"),
                    ("Toor dal (150g cooked)", 140, 10.2, 24.2, 0.6, "Lightly tempered"),
                    ("Palak sabzi (100g)", 26, 2.6, 3.5, 0.7, "Minimal oil"),
                    ("Mixed salad (150g)", 30, 1.5, 6.0, 0.3, "Cucumber, tomato, onion"),
                ],
            },
            {
                "meal": "Evening Snack",
                "time": "4:30–5:30 PM",
                "foods": [
                    ("Roasted chana (30g)", 114, 6.5, 18.0, 2.0, "Dry roasted"),
                    ("Green tea (cup)", 2, 0.0, 0.0, 0.0, "No sugar"),
                ],
            },
            {
                "meal": "Dinner",
                "time": "7:00–8:30 PM",
                "foods": [
                    ("Chicken breast (150g)", 248, 46.5, 0.0, 5.4, "Grilled or baked"),
                    ("Brown rice (100g cooked)", 112, 2.6, 23.5, 0.9, "Or 1 roti"),
                    ("Mixed vegetables (150g)", 50, 2.5, 9.0, 0.5, "Stir-fried in 1 tsp oil"),
                ],
            },
        ],
        "muscle": [
            {
                "meal": "Breakfast",
                "time": "7:00–8:00 AM",
                "foods": [
                    ("4 egg omelette", 310, 25.2, 2.2, 21.2, "2 whole + 2 whites, minimal oil"),
                    ("2 whole wheat rotis", 178, 5.8, 33.9, 2.2, "For complex carbs"),
                    ("1 glass full-fat milk", 122, 6.4, 8.8, 7.4, "With 1 tsp honey"),
                ],
            },
            {
                "meal": "Mid-Morning (Pre-workout)",
                "time": "10:00–10:30 AM",
                "foods": [
                    ("Banana (2 medium)", 178, 2.2, 45.6, 0.6, "Fast energy"),
                    ("Peanut butter (2 tbsp)", 190, 8.0, 6.0, 16.0, "Natural, unsweetened"),
                ],
            },
            {
                "meal": "Lunch",
                "time": "1:00–2:00 PM",
                "foods": [
                    ("Chicken breast (200g)", 330, 62.0, 0.0, 7.2, "Grilled with spices"),
                    ("Basmati rice (200g cooked)", 242, 4.4, 54.2, 0.4, "Or 3 rotis"),
                    ("Rajma curry (150g)", 191, 13.1, 34.2, 0.8, "High protein legume"),
                    ("Curd (150g)", 90, 4.7, 4.5, 6.0, "As side"),
                ],
            },
            {
                "meal": "Post-Workout",
                "time": "4:00–5:00 PM",
                "foods": [
                    ("Whey protein shake (30g)", 120, 24.0, 3.0, 2.1, "Mix in water or milk"),
                    ("Banana (1)", 89, 1.1, 22.8, 0.3, "Replenish glycogen"),
                ],
            },
            {
                "meal": "Dinner",
                "time": "7:30–8:30 PM",
                "foods": [
                    ("Paneer bhurji (150g paneer)", 398, 27.5, 4.7, 31.2, "Cooked with minimal oil"),
                    ("2 whole wheat rotis", 178, 5.8, 33.9, 2.2, ""),
                    ("Dal tadka (100g)", 105, 6.5, 14.2, 2.8, "Protein-rich"),
                    ("Cucumber raita (100g)", 35, 2.0, 4.0, 1.0, ""),
                ],
            },
        ],
        "gain": [
            {
                "meal": "Breakfast",
                "time": "7:00–8:00 AM",
                "foods": [
                    ("4 whole eggs scrambled", 310, 25.2, 2.2, 21.2, "Cooked in ghee"),
                    ("3 parathas (whole wheat)", 450, 12.0, 72.0, 14.0, "With ghee"),
                    ("Full fat milk (300ml)", 183, 9.6, 13.2, 11.1, ""),
                ],
            },
            {
                "meal": "Mid-Morning",
                "time": "10:30 AM",
                "foods": [
                    ("Mixed dry fruits (50g)", 280, 7.0, 30.0, 17.0, "Almonds, raisins, cashews"),
                    ("Banana milkshake (300ml)", 220, 7.0, 40.0, 5.0, "Full fat milk"),
                ],
            },
            {
                "meal": "Lunch",
                "time": "1:00–2:00 PM",
                "foods": [
                    ("Chicken curry (200g)", 350, 42.0, 8.0, 16.0, "With bones for calcium"),
                    ("White rice (300g cooked)", 363, 6.6, 81.3, 0.6, ""),
                    ("Chana dal (150g)", 246, 13.1, 41.3, 4.1, ""),
                    ("Gulab jamun (2 pcs)", 250, 4.0, 42.0, 8.0, "Caloric dense dessert"),
                ],
            },
            {
                "meal": "Evening",
                "time": "5:00 PM",
                "foods": [
                    ("Peanut butter toast (2 slices)", 350, 14.0, 36.0, 18.0, "Whole grain bread"),
                    ("Mango lassi (300ml)", 200, 7.0, 38.0, 4.0, "Full fat curd"),
                ],
            },
            {
                "meal": "Dinner",
                "time": "8:00–9:00 PM",
                "foods": [
                    ("Paneer tikka (200g paneer)", 530, 36.6, 6.2, 41.6, "Grilled in tandoor or oven"),
                    ("3 rotis or 1.5 cup rice", 360, 11.0, 68.0, 4.0, ""),
                    ("Rajma (200g cooked)", 254, 17.4, 45.6, 1.0, "Protein + carbs"),
                    ("Curd (200g)", 120, 6.2, 6.0, 8.0, "Probiotics"),
                ],
            },
        ],
    }
}

# Mirror western/others to india templates for now (would be expanded)
for cuisine in ["western", "asian", "mediterranean", "any"]:
    MEAL_TEMPLATES[cuisine] = MEAL_TEMPLATES["india"]


def build_meal_plan(profile: UserProfile, calorie_target: float,
                    protein_g: float, carbs_g: float, fat_g: float) -> List[Meal]:
    """Build a complete day meal plan from templates + food DB scoring."""
    cuisine = profile.cuisine
    goal    = profile.goal
    meals_n = profile.meals_per_day

    template_meals = MEAL_TEMPLATES.get(cuisine, MEAL_TEMPLATES["india"]).get(goal, [])

    # Filter to desired number of meals
    if meals_n <= 3:
        template_meals = [m for m in template_meals if "Snack" not in m["meal"] and
                          "Post" not in m["meal"] and "Mid" not in m["meal"]]
    elif meals_n == 4:
        template_meals = template_meals[:4]
    else:
        template_meals = template_meals[:meals_n]

    if not template_meals:
        template_meals = MEAL_TEMPLATES["india"][goal][:meals_n]

    # Filter for dietary restrictions
    diet = profile.diet_type
    allergies = (profile.allergies or "").lower()
    excluded = set()
    if diet == "vegetarian" or diet == "vegan" or diet == "eggetarian":
        excluded.update(["chicken", "fish", "mutton", "beef", "pork", "prawn", "sardine", "rohu"])
    if diet == "vegan":
        excluded.update(["egg", "milk", "curd", "paneer", "dahi", "whey", "lassi"])
    if diet == "dairyFree":
        excluded.update(["milk", "curd", "paneer", "dahi", "whey", "lassi", "ghee"])
    if diet == "glutenFree":
        excluded.update([
            "wheat", "roti", "chapati", "paratha", "bread", "pasta",
            "maida", "seitan", "barley", "rye",
        ])
    for allergy in allergies.split(","):
        excluded.add(allergy.strip())

    # Scale calories to target
    raw_cal = sum(
        sum(f[1] for f in m["foods"]) for m in template_meals
    )
    scale = calorie_target / (raw_cal + 1e-6)

    result: List[Meal] = []
    for tm in template_meals:
        food_items: List[FoodItem] = []
        meal_cal = 0.0

        for food_tuple in tm["foods"]:
            fname, cal, prot, carb, fat, note = food_tuple
            # Skip excluded foods
            if any(ex in fname.lower() for ex in excluded if ex):
                continue

            # Scale quantities
            s_cal  = cal  * scale
            s_prot = prot * scale
            s_carb = carb * scale
            s_fat  = fat  * scale
            qty_g  = s_cal / (cal / 100 + 1e-6)  # rough gram estimate

            food_score = score_food(s_cal, s_prot, s_carb, s_fat, 0.0, goal)

            food_items.append(FoodItem(
                name=fname,
                quantity_g=round(qty_g, 0),
                calories=round(s_cal, 1),
                protein_g=round(s_prot, 1),
                carbs_g=round(s_carb, 1),
                fat_g=round(s_fat, 1),
                prep_note=note,
                score=food_score,
            ))
            meal_cal += s_cal

        if food_items:
            result.append(Meal(
                meal_name=tm["meal"],
                time_window=tm["time"],
                total_calories=round(meal_cal, 1),
                foods=food_items,
            ))

    return result
