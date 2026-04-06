"""
backend/ml_engine.py
────────────────────
Loads trained TensorFlow models and exposes clean Python functions
for the FastAPI routers to call.

Falls back to analytical formulas (Mifflin-St Jeor) if models
haven't been trained yet — so the API works immediately out of the box.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import joblib

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_tdee_model    = None
_macro_model   = None
_scorer_model  = None
_tdee_scaler   = None
_macro_scaler_X = None
_macro_scaler_y = None
_scorer_scaler  = None
_le_sex         = None
_le_activity    = None
_le_goal        = None
_models_loaded  = False


def _load_models():
    global _tdee_model, _macro_model, _scorer_model
    global _tdee_scaler, _macro_scaler_X, _macro_scaler_y, _scorer_scaler
    global _le_sex, _le_activity, _le_goal, _models_loaded

    if _models_loaded:
        return

    try:
        import tensorflow as tf
        if (MODEL_DIR / "tdee_model.keras").exists():
            _tdee_model  = tf.keras.models.load_model(MODEL_DIR / "tdee_model.keras", compile=False)
            _macro_model = tf.keras.models.load_model(MODEL_DIR / "macro_model.keras", compile=False)
            _scorer_model = tf.keras.models.load_model(MODEL_DIR / "scorer_model.keras", compile=False)
            _tdee_scaler  = joblib.load(MODEL_DIR / "tdee_scaler.pkl")
            _macro_scaler_X = joblib.load(MODEL_DIR / "macro_scaler_X.pkl")
            _macro_scaler_y = joblib.load(MODEL_DIR / "macro_scaler_y.pkl")
            _scorer_scaler  = joblib.load(MODEL_DIR / "scorer_scaler.pkl")
            _le_sex         = joblib.load(MODEL_DIR / "le_sex.pkl")
            _le_activity    = joblib.load(MODEL_DIR / "le_activity.pkl")
            _le_goal        = joblib.load(MODEL_DIR / "le_goal.pkl")
            log.info("✅ TensorFlow models loaded from backend/models/")
        else:
            log.warning("⚠ TF models not found — using analytical fallback formulas")
    except Exception as e:
        log.warning(f"⚠ Could not load TF models ({e}) — using analytical fallback")

    _models_loaded = True


# ─── Activity factor map ──────────────────────────────────────────────────────

ACTIVITY_FACTORS = {
    "sedentary": 1.20,
    "light":     1.375,
    "moderate":  1.55,
    "very":      1.725,
    "extreme":   1.90,
}

GOAL_LABELS = {
    "loss":   "Fat Loss",
    "muscle": "Muscle Gain",
    "gain":   "Healthy Bulk",
}


# ─── BMI ──────────────────────────────────────────────────────────────────────

def compute_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25.0:
        cat = "Normal weight"
    elif bmi < 30.0:
        cat = "Overweight"
    else:
        cat = "Obese"
    return round(bmi, 1), cat


# ─── TDEE ─────────────────────────────────────────────────────────────────────

def predict_tdee(
    age: int, sex: str, height_cm: float, weight_kg: float,
    body_fat_pct: Optional[float], activity_level: str
) -> tuple[float, float]:
    """
    Returns (BMR, TDEE) in kcal.
    Uses TF model if available, otherwise Mifflin-St Jeor / Katch-McArdle.
    """
    _load_models()

    # Analytical BMR (always compute as ground-truth anchor)
    if body_fat_pct:
        lbm = weight_kg * (1 - body_fat_pct / 100)
        bmr_analytical = 370 + 21.6 * lbm          # Katch-McArdle
    else:
        if sex == "male":
            bmr_analytical = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr_analytical = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    tdee_analytical = bmr_analytical * ACTIVITY_FACTORS[activity_level]

    if _tdee_model is not None:
        try:
            bmi = weight_kg / (height_cm / 100) ** 2
            sex_enc = _le_sex.transform([sex])[0]
            act_enc = _le_activity.transform([activity_level])[0]
            X = np.array([[age, sex_enc, height_cm, weight_kg,
                           body_fat_pct or bmi, bmi, act_enc, 7.0]], dtype="float32")
            X_s = _tdee_scaler.transform(X)
            tdee_pred = float(_tdee_model.predict(X_s, verbose=0)[0][0])
            # Blend TF prediction (60%) with analytical (40%) for robustness
            tdee = 0.60 * tdee_pred + 0.40 * tdee_analytical
        except Exception as e:
            log.warning(f"TF TDEE inference failed ({e}), using analytical")
            tdee = tdee_analytical
    else:
        tdee = tdee_analytical

    return round(bmr_analytical), round(tdee)


# ─── Macros ───────────────────────────────────────────────────────────────────

def predict_macros(
    tdee: float, calorie_target: float, goal: str,
    bmi: float, body_fat_pct: float, activity_level: str,
    sleep_h: float, age: int
) -> tuple[float, float, float]:
    """
    Returns (protein_g, carbs_g, fat_g).
    """
    _load_models()

    # Analytical fallback
    if goal == "loss":
        p_pct, c_pct, f_pct = 0.35, 0.35, 0.30
    elif goal == "muscle":
        p_pct, c_pct, f_pct = 0.35, 0.45, 0.20
    else:  # gain
        p_pct, c_pct, f_pct = 0.20, 0.55, 0.25

    p_ana = calorie_target * p_pct / 4
    c_ana = calorie_target * c_pct / 4
    f_ana = calorie_target * f_pct / 9

    if _macro_model is not None:
        try:
            goal_enc = _le_goal.transform([goal])[0]
            act_enc  = _le_activity.transform([activity_level])[0]
            X = np.array([[tdee, calorie_target, goal_enc, bmi,
                           body_fat_pct, act_enc, sleep_h, age]], dtype="float32")
            X_s = _macro_scaler_X.transform(X)
            y_s = _macro_model.predict(X_s, verbose=0)
            y   = _macro_scaler_y.inverse_transform(y_s)[0]
            protein_g, carbs_g, fat_g = float(y[0]), float(y[1]), float(y[2])
            # Sanity clip
            protein_g = max(min(protein_g, calorie_target * 0.50 / 4), 50)
            carbs_g   = max(min(carbs_g,   calorie_target * 0.70 / 4), 20)
            fat_g     = max(min(fat_g,     calorie_target * 0.50 / 9), 20)
            # Blend
            protein_g = round(0.6 * protein_g + 0.4 * p_ana, 1)
            carbs_g   = round(0.6 * carbs_g   + 0.4 * c_ana, 1)
            fat_g     = round(0.6 * fat_g     + 0.4 * f_ana, 1)
        except Exception as e:
            log.warning(f"TF macro inference failed ({e}), using analytical")
            protein_g, carbs_g, fat_g = round(p_ana, 1), round(c_ana, 1), round(f_ana, 1)
    else:
        protein_g, carbs_g, fat_g = round(p_ana, 1), round(c_ana, 1), round(f_ana, 1)

    return protein_g, carbs_g, fat_g


# ─── Food Score ───────────────────────────────────────────────────────────────

def score_food(
    calories: float, protein_g: float, carbs_g: float,
    fat_g: float, fiber_g: float, goal: str
) -> float:
    """Returns suitability score 0–100 for this food + goal combo."""
    _load_models()

    goal_map = {"loss": 0, "muscle": 1, "gain": 2}
    g_idx = goal_map.get(goal, 0)

    total_macro = protein_g * 4 + carbs_g * 4 + fat_g * 9 + 0.001
    prot_density = (protein_g * 4) / total_macro
    carb_density = (carbs_g * 4) / total_macro
    fat_density  = (fat_g * 9) / total_macro

    # Analytical score
    if goal == "loss":
        score = (40 * prot_density + 20 * (fiber_g / (fiber_g + 1)) +
                 20 * (1 - min(calories, 500) / 500) + 20 * (1 - fat_density * 0.5))
    elif goal == "muscle":
        score = (50 * prot_density + 25 * carb_density +
                 15 * (fiber_g / (fiber_g + 1)) + 10 * (1 - fat_density * 0.3))
    else:
        score = (30 * prot_density + 35 * carb_density +
                 20 * (min(calories, 600) / 600) + 15 * (1 - fat_density * 0.2))
    score_analytical = min(max(score * 100, 0), 100)

    if _scorer_model is not None:
        try:
            X = np.array([[calories, protein_g, carbs_g, fat_g, fiber_g,
                           prot_density, carb_density, fat_density, g_idx]], dtype="float32")
            X_s = _scorer_scaler.transform(X)
            score_tf = float(_scorer_model.predict(X_s, verbose=0)[0][0]) * 100
            return round(0.6 * score_tf + 0.4 * score_analytical, 1)
        except Exception:
            pass

    return round(score_analytical, 1)
