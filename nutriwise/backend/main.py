"""
backend/main.py
───────────────
FastAPI application — NutriWise backend.

Endpoints:
  POST /api/plan          → generate full nutrition plan
  GET  /api/foods/search  → search food database
  GET  /api/foods/score   → score a food for a given goal
  GET  /api/plots/{name}  → serve training plots
  GET  /                  → serve frontend HTML
  GET  /health            → healthcheck
"""

from __future__ import annotations
import logging, math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    UserProfile, NutritionPlan, MacroBreakdown,
    WeeklyProjection, FoodSearchResult
)
from backend.ml_engine import (
    predict_tdee, predict_macros, score_food,
    compute_bmi, GOAL_LABELS, ACTIVITY_FACTORS
)
from backend.utils.meal_builder import build_meal_plan

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

app = FastAPI(
    title="NutriWise API",
    description="AI-powered personalized nutrition planning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = ROOT / "frontend" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

plots_path = ROOT / "backend" / "plots"
if plots_path.exists():
    app.mount("/plots", StaticFiles(directory=str(plots_path)), name="plots")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _generate_tips(profile: UserProfile, calorie_target: float, protein_g: float) -> list[str]:
    tips = []
    goal = profile.goal

    if goal == "loss":
        tips += [
            f"Eat in a calorie deficit of ~{int(profile.weight_kg * 7.7 * 0.5)} kcal/week for sustainable fat loss.",
            "Prioritize protein at every meal to preserve muscle while cutting.",
            "Eat vegetables first — their fiber slows glucose absorption and increases satiety.",
            "Avoid liquid calories (juices, sodas, chai with sugar) — they add up fast.",
            "Do 30–45 min of cardio 4–5 days/week alongside strength training.",
        ]
    elif goal == "muscle":
        tips += [
            f"Aim for {round(profile.weight_kg * 1.6, 0)}–{round(profile.weight_kg * 2.2, 0)}g protein/day for optimal muscle synthesis.",
            "Eat a protein + carb meal within 30–60 minutes after your workout.",
            "Progressive overload in your training matters more than any supplement.",
            "Get 7–9 hours of sleep — 70% of muscle repair happens during deep sleep.",
            "Track your lifts weekly; if you aren't getting stronger, eat slightly more.",
        ]
    else:
        tips += [
            "Eat every 3–4 hours to maintain a consistent caloric surplus.",
            "Include calorie-dense foods like nuts, ghee, avocado, and peanut butter.",
            "Don't skip meals — missing one meal can put you in a deficit for the day.",
            "Strength train 4–5 days/week to ensure gained weight is lean mass.",
            "Liquid calories (milk, smoothies) are your best friend for hitting calorie targets.",
        ]

    # Universal
    tips += [
        f"Drink {round(profile.weight_kg * 35)} ml of water daily ({round(profile.weight_kg * 35 / 1000, 1)}L).",
        "Meal-prep on Sundays to avoid poor food choices on busy days.",
    ]
    if profile.health_notes:
        tips.append(f"Given '{profile.health_notes}' — consult a registered dietitian for personalized medical guidance.")

    return tips[:7]


def _generate_supplements(profile: UserProfile) -> list[str]:
    sups = []
    goal = profile.goal

    # Universal
    sups.append("Vitamin D3 (2000–4000 IU/day) — most Indians are deficient")
    sups.append("Omega-3 fish oil (1–3g EPA+DHA/day) — anti-inflammatory, heart health")

    if goal in ("muscle", "gain"):
        sups += [
            "Creatine monohydrate (5g/day) — most studied, safe performance booster",
            "Whey protein (20–30g post-workout) if whole food protein is insufficient",
        ]
    if goal == "loss":
        sups.append("Magnesium glycinate (200–400mg at night) — improves sleep and fat oxidation")

    if profile.diet_type in ("vegetarian", "vegan"):
        sups.append("Vitamin B12 (500–1000 mcg/day) — essential, absent in plant foods")

    return sups


def _generate_warnings(profile: UserProfile, bmi: float, calorie_target: float) -> list[str]:
    warnings = []
    if calorie_target < 1200:
        warnings.append("⚠ Calorie target below 1200 kcal — risk of nutrient deficiency. Consult a dietitian.")
    if bmi > 35:
        warnings.append("⚠ BMI > 35 — medical supervision recommended before starting any diet plan.")
    if bmi < 17:
        warnings.append("⚠ BMI < 17 — you may be severely underweight. Please consult a doctor.")
    if profile.health_notes:
        h = profile.health_notes.lower()
        if any(x in h for x in ["diabetes", "diabetic"]):
            warnings.append("⚠ Diabetes detected — monitor carbohydrate intake carefully and consult an endocrinologist.")
        if any(x in h for x in ["thyroid"]):
            warnings.append("⚠ Thyroid condition — calorie needs may differ; get TSH checked regularly.")
        if any(x in h for x in ["pcod", "pcos"]):
            warnings.append("⚠ PCOD/PCOS — low-GI carbs and reduced sugar are particularly important.")
    return warnings


def _project_weekly(
    profile: UserProfile,
    calorie_target: float, tdee: float,
    protein_g: float, carbs_g: float, fat_g: float,
    weeks: int = 8
) -> list[WeeklyProjection]:
    """Project weekly metrics over the plan timeline."""
    projections = []
    weight = profile.weight_kg
    deficit_per_week = (tdee - calorie_target) * 7  # kcal/week
    kg_per_week = deficit_per_week / 7700            # ~7700 kcal = 1 kg fat

    for w in range(1, weeks + 1):
        weekly_cal = calorie_target
        weekly_prot = protein_g
        weekly_carbs = carbs_g
        weekly_fat = fat_g

        # Progressive adjustment (small weekly tweaks)
        if profile.goal == "loss":
            projected_w = max(weight - kg_per_week * w, profile.target_weight_kg or (weight - 20))
        elif profile.goal in ("muscle", "gain"):
            projected_w = weight + abs(kg_per_week) * w * 0.5  # slower gain
        else:
            projected_w = weight

        projections.append(WeeklyProjection(
            week=w,
            calorie_target=round(weekly_cal),
            protein_g=round(weekly_prot, 1),
            carbs_g=round(weekly_carbs, 1),
            fat_g=round(weekly_fat, 1),
            projected_weight_kg=round(projected_w, 1),
        ))

    return projections


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = ROOT / "frontend" / "templates" / "index.html"
    if html_path.exists():
        # Force UTF-8 to avoid mojibake on Windows locale defaults.
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>NutriWise API</h1><p>Frontend not built yet.</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/plan", response_model=NutritionPlan)
async def generate_plan(profile: UserProfile):
    """Generate a full personalized nutrition plan."""
    log.info(f"Plan request: {profile.goal} / {profile.sex} / {profile.age}y / {profile.weight_kg}kg")

    try:
        bmi, bmi_cat = compute_bmi(profile.weight_kg, profile.height_cm)

        # ML predictions
        bmr, tdee = predict_tdee(
            age=profile.age, sex=profile.sex,
            height_cm=profile.height_cm, weight_kg=profile.weight_kg,
            body_fat_pct=profile.body_fat_pct,
            activity_level=profile.activity_level,
        )

        # Goal-adjusted calorie target
        if profile.goal == "loss":
            calorie_target = max(1200, tdee - 400)
        elif profile.goal == "muscle":
            calorie_target = tdee + 250
        else:  # gain
            calorie_target = tdee + 350

        protein_g, carbs_g, fat_g = predict_macros(
            tdee=tdee, calorie_target=calorie_target,
            goal=profile.goal, bmi=bmi,
            body_fat_pct=profile.body_fat_pct or bmi,
            activity_level=profile.activity_level,
            sleep_h=profile.sleep_h, age=profile.age,
        )

        total_macro_kcal = protein_g * 4 + carbs_g * 4 + fat_g * 9
        macros = MacroBreakdown(
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            protein_pct=round(protein_g * 4 / total_macro_kcal * 100, 1),
            carbs_pct=round(carbs_g * 4 / total_macro_kcal * 100, 1),
            fat_pct=round(fat_g * 9 / total_macro_kcal * 100, 1),
        )

        meals = build_meal_plan(profile, calorie_target, protein_g, carbs_g, fat_g)

        weeks = profile.timeline_weeks or 12
        weekly_projections = _project_weekly(
            profile, calorie_target, tdee,
            protein_g, carbs_g, fat_g, weeks=min(weeks, 12)
        )

        return NutritionPlan(
            bmr=bmr,
            tdee=tdee,
            calorie_target=round(calorie_target),
            bmi=bmi,
            bmi_category=bmi_cat,
            macros=macros,
            meals=meals,
            weekly_projections=weekly_projections,
            hydration_ml=round(profile.weight_kg * 35),
            tips=_generate_tips(profile, calorie_target, protein_g),
            supplements=_generate_supplements(profile),
            warnings=_generate_warnings(profile, bmi, calorie_target),
            goal_code=profile.goal,
            goal_label=GOAL_LABELS.get(profile.goal, "Custom"),
        )

    except Exception as e:
        log.error(f"Plan generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/foods/search", response_model=list[FoodSearchResult])
async def search_foods(
    q: str = Query(..., min_length=2),
    goal: str = Query("muscle"),
    limit: int = Query(20, le=100),
):
    """Search the food database and return scored results."""
    from backend.utils.meal_builder import _load_food_db
    df = _load_food_db()
    if df.empty:
        return []

    mask = df["name"].str.lower().str.contains(q.lower(), na=False)
    results = df[mask].head(limit)

    out = []
    for _, row in results.iterrows():
        cal   = float(row.get("calories", 0) or 0)
        prot  = float(row.get("protein_g", 0) or 0)
        carb  = float(row.get("carbs_g", 0) or 0)
        fat   = float(row.get("fat_g", 0) or 0)
        fiber = float(row.get("fiber_g", 0) or 0)
        out.append(FoodSearchResult(
            fdc_id=int(row["fdc_id"]) if "fdc_id" in row and not math.isnan(float(row.get("fdc_id", 0) or 0)) else None,
            name=str(row["name"]),
            category=str(row.get("category", "")),
            calories=cal,
            protein_g=prot,
            carbs_g=carb,
            fat_g=fat,
            fiber_g=fiber,
            score=score_food(cal, prot, carb, fat, fiber, goal),
            source=str(row.get("source", "USDA")),
        ))

    out.sort(key=lambda x: x.score or 0, reverse=True)
    return out


@app.get("/api/foods/score")
async def get_food_score(
    calories: float = Query(...),
    protein_g: float = Query(...),
    carbs_g: float = Query(...),
    fat_g: float = Query(...),
    fiber_g: float = Query(0),
    goal: str = Query("muscle"),
):
    """Score a custom food for a given goal."""
    s = score_food(calories, protein_g, carbs_g, fat_g, fiber_g, goal)
    return {"score": s, "goal": goal}


@app.get("/api/plots")
async def list_plots():
    """List available training visualization plots."""
    if not plots_path.exists():
        return {"plots": []}
    return {"plots": [f.name for f in plots_path.glob("*.png")]}
