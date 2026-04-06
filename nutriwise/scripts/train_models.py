"""
scripts/train_models.py
───────────────────────
Trains three TensorFlow/Keras models:

  Model 1 — TDEE Regressor
    Input : age, sex, height, weight, body_fat, activity_level
    Output: daily TDEE (kcal)

  Model 2 — Macro Recommender
    Input : TDEE, goal, bmi, activity_level, sleep
    Output: protein_g, carbs_g, fat_g (per day)

  Model 3 — Food Scorer
    Input : food macros + user goal
    Output: suitability score 0–100

All models are saved to backend/models/ as SavedModel format.
Scaler artifacts are saved with joblib for inference-time use.

Run:
    python scripts/train_models.py
"""

import os, sys, platform, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
except ModuleNotFoundError as e:
    if e.name == "tensorflow":
        py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print("\n[ERROR] TensorFlow is not installed (or unavailable for this Python version).")
        print(f"Detected Python: {py} on {platform.system()}")
        print("NutriWise training requires Python 3.10 or 3.11.")
        print("\nFix:")
        print("  1) Install Python 3.11")
        print("  2) Create and activate a fresh venv with that Python")
        print("  3) Reinstall deps: pip install -r requirements.txt")
        print("  4) Re-run: python scripts/train_models.py")
        raise SystemExit(1)
    raise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

print(f"TensorFlow {tf.__version__} — GPU: {bool(tf.config.list_physical_devices('GPU'))}")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "backend" / "data"
MODEL_DIR = ROOT / "backend" / "models"
PLOT_DIR  = ROOT / "backend" / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_plot(fig, name: str):
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot saved → {path.name}")


def plot_training(history, title: str, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"], label="train", color="#2D6A4F")
    axes[0].plot(history.history["val_loss"], label="val", color="#F4A261")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metric_key = [k for k in history.history if "mae" in k and "val" not in k]
    if metric_key:
        key = metric_key[0]
        axes[1].plot(history.history[key], label="train", color="#2D6A4F")
        axes[1].plot(history.history[f"val_{key}"], label="val", color="#F4A261")
        axes[1].set_title("MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    save_plot(fig, name)


def plot_predictions(y_true, y_pred, label: str, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{label} — Prediction Quality", fontsize=13, fontweight="bold")

    axes[0].scatter(y_true, y_pred, alpha=0.25, s=8, color="#2D6A4F")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")
    axes[0].grid(alpha=0.3)

    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, color="#52B788", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", lw=1.5, ls="--")
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Error (predicted − actual)")
    axes[1].grid(alpha=0.3)

    save_plot(fig, name)


# ─── Model 1: TDEE Regressor ──────────────────────────────────────────────────

def train_tdee_model(df: pd.DataFrame):
    print("\n" + "═" * 50)
    print("  Model 1 — TDEE Regressor")
    print("═" * 50)

    # Encode
    le_sex      = LabelEncoder().fit(df["sex"])
    le_activity = LabelEncoder().fit(df["activity_level"])
    joblib.dump(le_sex,      MODEL_DIR / "le_sex.pkl")
    joblib.dump(le_activity, MODEL_DIR / "le_activity.pkl")

    X = pd.DataFrame({
        "age":            df["age"],
        "sex":            le_sex.transform(df["sex"]),
        "height_cm":      df["height_cm"],
        "weight_kg":      df["weight_kg"],
        "body_fat_pct":   df["body_fat_pct"],
        "bmi":            df["bmi"],
        "activity_level": le_activity.transform(df["activity_level"]),
        "sleep_h":        df["sleep_h"],
    }).values.astype("float32")

    y = df["tdee"].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "tdee_scaler.pkl")

    # Architecture
    model = keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.10),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear"),   # kcal output
    ], name="tdee_regressor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=256,
        callbacks=cb,
        verbose=1,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n  Test MAE  : {mae:.1f} kcal")
    print(f"  Test R²   : {r2:.4f}")
    print(f"  Test MAPE : {mape:.2f}%")

    model.save(MODEL_DIR / "tdee_model.keras")
    print(f"  ✅ Saved → backend/models/tdee_model.keras")

    plot_training(history, "Model 1 — TDEE Regressor", "tdee_training")
    plot_predictions(y_test, y_pred, "TDEE (kcal)", "tdee_predictions")

    return model, scaler


# ─── Model 2: Macro Recommender ───────────────────────────────────────────────

def train_macro_model(df: pd.DataFrame):
    print("\n" + "═" * 50)
    print("  Model 2 — Macro Recommender")
    print("═" * 50)

    le_goal     = LabelEncoder().fit(df["goal"])
    le_activity = joblib.load(MODEL_DIR / "le_activity.pkl")
    joblib.dump(le_goal, MODEL_DIR / "le_goal.pkl")

    X = pd.DataFrame({
        "tdee":           df["tdee"],
        "calorie_target": df["calorie_target"],
        "goal":           le_goal.transform(df["goal"]),
        "bmi":            df["bmi"],
        "body_fat_pct":   df["body_fat_pct"],
        "activity_level": le_activity.transform(df["activity_level"]),
        "sleep_h":        df["sleep_h"],
        "age":            df["age"],
    }).values.astype("float32")

    y = df[["protein_g", "carbs_g", "fat_g"]].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)

    joblib.dump(scaler_X, MODEL_DIR / "macro_scaler_X.pkl")
    joblib.dump(scaler_y, MODEL_DIR / "macro_scaler_y.pkl")

    model = keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.10),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="linear"),   # [protein, carbs, fat]
    ], name="macro_recommender")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train_s,
        validation_split=0.15,
        epochs=200,
        batch_size=256,
        callbacks=cb,
        verbose=1,
    )

    y_pred_s = model.predict(X_test, verbose=0)
    y_pred   = scaler_y.inverse_transform(y_pred_s)

    macro_names = ["Protein (g)", "Carbs (g)", "Fat (g)"]
    for i, name in enumerate(macro_names):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / (y_test[:, i] + 1e-6))) * 100
        print(f"  {name:15s} → MAE {mae:.1f}g  MAPE {mape:.2f}%")

    model.save(MODEL_DIR / "macro_model.keras")
    print(f"  ✅ Saved → backend/models/macro_model.keras")

    plot_training(history, "Model 2 — Macro Recommender", "macro_training")

    # Plot each macro
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Macro Predictions vs Actual", fontsize=13, fontweight="bold")
    colors = ["#2D6A4F", "#F4A261", "#4A90D9"]
    for i, (name, color) in enumerate(zip(macro_names, colors)):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.2, s=6, color=color)
        lim = [0, max(y_test[:, i].max(), y_pred[:, i].max())]
        axes[i].plot(lim, lim, "r--", lw=1)
        axes[i].set_title(name)
        axes[i].set_xlabel("Actual")
        axes[i].grid(alpha=0.3)
    save_plot(fig, "macro_predictions")

    return model


# ─── Model 3: Food Scorer ─────────────────────────────────────────────────────

def train_food_scorer(food_df: pd.DataFrame, nhanes_df: pd.DataFrame):
    """
    Score any food's suitability (0–100) for a given goal.
    Uses rule-based scoring to generate labels, then trains a small NN
    so it can generalize to unseen foods.
    """
    print("\n" + "═" * 50)
    print("  Model 3 — Food Scorer")
    print("═" * 50)

    goals = ["loss", "muscle", "gain"]
    rows = []

    for _, food in food_df.iterrows():
        cal   = float(food.get("calories", 0) or 0)
        prot  = float(food.get("protein_g", 0) or 0)
        carb  = float(food.get("carbs_g", 0) or 0)
        fat   = float(food.get("fat_g", 0) or 0)
        fiber = float(food.get("fiber_g", 0) or 0)
        total_macro = prot * 4 + carb * 4 + fat * 9 + 0.001

        for g_idx, goal in enumerate(goals):
            # Rule-based score (label) per 100g
            prot_density = (prot * 4) / total_macro
            carb_density = (carb * 4) / total_macro
            fat_density  = (fat * 9) / total_macro

            if goal == "loss":
                score = (
                    40 * prot_density +
                    20 * (fiber / (fiber + 1)) +
                    20 * (1 - min(cal, 500) / 500) +
                    20 * (1 - fat_density * 0.5)
                )
            elif goal == "muscle":
                score = (
                    50 * prot_density +
                    25 * carb_density +
                    15 * (fiber / (fiber + 1)) +
                    10 * (1 - fat_density * 0.3)
                )
            else:  # gain
                score = (
                    30 * prot_density +
                    35 * carb_density +
                    20 * (min(cal, 600) / 600) +
                    15 * (1 - fat_density * 0.2)
                )

            rows.append({
                "calories": cal,
                "protein_g": prot,
                "carbs_g": carb,
                "fat_g": fat,
                "fiber_g": fiber,
                "prot_density": prot_density,
                "carb_density": carb_density,
                "fat_density": fat_density,
                "goal_idx": g_idx,
                "score": min(max(score * 100, 0), 100),
            })

    scorer_df = pd.DataFrame(rows)

    feature_cols = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
                    "prot_density", "carb_density", "fat_density", "goal_idx"]
    X = scorer_df[feature_cols].values.astype("float32")
    y = scorer_df["score"].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scorer_scaler.pkl")

    model = keras.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="food_scorer")

    # Scale output to 0-100 via sigmoid * 100
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=lambda y_t, y_p: keras.losses.mse(y_t / 100, y_p),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )

    history = model.fit(
        X_train, y_train / 100,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )

    y_pred = model.predict(X_test, verbose=0).flatten() * 100
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n  Test MAE: {mae:.2f} score points")

    model.save(MODEL_DIR / "scorer_model.keras")
    print(f"  ✅ Saved → backend/models/scorer_model.keras")
    plot_training(history, "Model 3 — Food Scorer", "scorer_training")

    return model


# ─── Feature Importance (SHAP-style permutation) ──────────────────────────────

def plot_feature_importance(model, X_test_scaled, feature_names: list, title: str, name: str):
    baseline = model.predict(X_test_scaled, verbose=0)
    if baseline.ndim > 1:
        baseline = baseline[:, 0]

    importances = []
    for i in range(X_test_scaled.shape[1]):
        X_perm = X_test_scaled.copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = model.predict(X_perm, verbose=0)
        if perm_pred.ndim > 1:
            perm_pred = perm_pred[:, 0]
        importances.append(np.mean(np.abs(perm_pred - baseline)))

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)
    colors = plt.cm.YlGn(np.linspace(0.4, 0.9, len(importances)))
    ax.barh(np.array(feature_names)[sorted_idx],
            np.array(importances)[sorted_idx],
            color=colors[sorted_idx])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Permutation importance (mean |Δ prediction|)")
    ax.grid(axis="x", alpha=0.3)
    save_plot(fig, name)


# ─── EDA Plots ────────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame):
    print("\n[EDA] Generating exploratory plots …")

    # TDEE distribution by goal
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("NHANES-style Dataset — EDA", fontsize=13, fontweight="bold")
    palette = {"loss": "#E76F51", "muscle": "#2D6A4F", "gain": "#4A90D9"}

    for i, (col, label) in enumerate([
        ("tdee", "TDEE (kcal)"),
        ("bmi", "BMI"),
        ("body_fat_pct", "Body Fat %"),
    ]):
        for goal in ["loss", "muscle", "gain"]:
            subset = df[df["goal"] == goal][col]
            axes[i].hist(subset, bins=40, alpha=0.5,
                         label=goal, color=palette[goal], edgecolor="none")
        axes[i].set_title(label)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    save_plot(fig, "eda_distributions")

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, ax=ax, cmap="YlGn", center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    save_plot(fig, "eda_correlation")

    print("  📊 EDA plots saved to backend/plots/")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise — Model Training Pipeline")
    print("=" * 60)

    # Load data
    nhanes_path = DATA_DIR / "nhanes_synthetic.csv"
    food_path   = DATA_DIR / "food_nutrients.csv"
    ifct_path   = DATA_DIR / "ifct_indian_foods.csv"

    if not nhanes_path.exists():
        raise FileNotFoundError("Run scripts/download_datasets.py first!")

    nhanes_df = pd.read_csv(nhanes_path)
    food_df   = pd.read_csv(food_path) if food_path.exists() else pd.DataFrame()
    ifct_df   = pd.read_csv(ifct_path) if ifct_path.exists() else pd.DataFrame()

    # Combine food databases
    if not food_df.empty and not ifct_df.empty:
        # Align columns
        for col in ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]:
            if col not in food_df.columns:
                food_df[col] = 0.0
            if col not in ifct_df.columns:
                ifct_df[col] = 0.0
        combined_food = pd.concat([
            food_df[["name", "category", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]],
            ifct_df[["name", "category", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]],
        ], ignore_index=True).dropna(subset=["calories"]).query("calories > 0")
    elif not ifct_df.empty:
        combined_food = ifct_df
    else:
        combined_food = food_df

    print(f"\n  Dataset sizes:")
    print(f"    NHANES (synthetic): {len(nhanes_df):,} rows")
    print(f"    Food database     : {len(combined_food):,} items")

    # EDA
    plot_eda(nhanes_df)

    # Train models
    tdee_model, tdee_scaler = train_tdee_model(nhanes_df)
    macro_model              = train_macro_model(nhanes_df)
    scorer_model             = train_food_scorer(combined_food, nhanes_df)

    # Feature importance for TDEE model
    le_sex      = joblib.load(MODEL_DIR / "le_sex.pkl")
    le_activity = joblib.load(MODEL_DIR / "le_activity.pkl")
    X_sample = pd.DataFrame({
        "age":            nhanes_df["age"].sample(1000, random_state=0),
        "sex":            le_sex.transform(nhanes_df["sex"].sample(1000, random_state=0)),
        "height_cm":      nhanes_df["height_cm"].sample(1000, random_state=0),
        "weight_kg":      nhanes_df["weight_kg"].sample(1000, random_state=0),
        "body_fat_pct":   nhanes_df["body_fat_pct"].sample(1000, random_state=0),
        "bmi":            nhanes_df["bmi"].sample(1000, random_state=0),
        "activity_level": le_activity.transform(nhanes_df["activity_level"].sample(1000, random_state=0)),
        "sleep_h":        nhanes_df["sleep_h"].sample(1000, random_state=0),
    }).values.astype("float32")
    X_scaled = tdee_scaler.transform(X_sample)
    plot_feature_importance(
        tdee_model, X_scaled,
        ["age", "sex", "height", "weight", "body_fat", "bmi", "activity", "sleep"],
        "TDEE Model — Feature Importance", "tdee_importance"
    )

    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print("  Models saved to: backend/models/")
    print("  Plots  saved to: backend/plots/")
    print("  Next step: python -m uvicorn backend.main:app --reload")
    print("=" * 60)
