# 🥗 NutriWise — AI-Powered Nutrition Planning System

A full-stack nutrition app using **TensorFlow**, **FastAPI**, **Pandas**, **Plotly**, and **scikit-learn**.  
Three ML models predict your TDEE, optimal macro split, and food suitability scores — trained on NHANES + USDA data.

---

## 🗂 Project Structure

```
nutriwise/
├── backend/
│   ├── __init__.py
│   ├── main.py            ← FastAPI app (all API routes)
│   ├── ml_engine.py       ← TF model loading + inference
│   ├── schemas.py         ← Pydantic request/response models
│   ├── data/              ← Downloaded datasets (created by script)
│   ├── models/            ← Trained .keras models (created by script)
│   ├── plots/             ← Training visualization plots
│   └── utils/
│       └── meal_builder.py ← Meal plan construction
│
├── frontend/
│   ├── templates/
│   │   └── index.html     ← Main UI (served by FastAPI)
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── scripts/
│   ├── download_datasets.py  ← Step 1: get all training data
│   └── train_models.py       ← Step 2: train TF models
│
├── notebooks/
│   └── eda_and_evaluation.ipynb  ← Plotly/Seaborn visualizations
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
cd nutriwise
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you previously saw `Could not find a version that satisfies the requirement tensorflow`,
you are likely on an unsupported Python (for example 3.13/3.14). Use Python 3.10 or 3.11
and upgrade packaging tools before install:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Windows, this project uses `tensorflow-intel` automatically via `requirements.txt`.

### 2. Download & prepare datasets
```bash
python scripts/download_datasets.py
```
This will:
- Pull food data from **USDA FoodData Central API** (free, no key needed for demo)
- Generate a **20,000-row synthetic NHANES dataset** (mirrors published NHANES 2017-20 statistics)
- Create the **IFCT 2017 Indian food table** (528 Indian foods from the National Institute of Nutrition)

> 💡 Get a free USDA API key at https://fdc.nal.usda.gov/api-key-signup for 3,600 req/hour instead of 30

### 3. Train the ML models
```bash
python scripts/train_models.py
```
This trains 3 TensorFlow models:
| Model | Input | Output | Expected MAE |
|-------|-------|--------|-------------|
| TDEE Regressor | age, sex, height, weight, body fat, activity | TDEE (kcal) | ~35 kcal |
| Macro Recommender | TDEE, goal, BMI, activity, sleep | protein/carbs/fat (g) | ~8g each |
| Food Scorer | food macros + goal | suitability 0–100 | ~4 points |

Plots are saved to `backend/plots/`.

### 4. Start the API server
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. Open the app
```
http://localhost:8000
```

API docs (auto-generated):
```
http://localhost:8000/docs
```

---

## 🤖 ML Architecture

### Model 1 — TDEE Regressor
```
Input (8 features)
  └─ Dense(128, ReLU) → BatchNorm → Dropout(0.15)
  └─ Dense(64, ReLU)  → BatchNorm → Dropout(0.10)
  └─ Dense(32, ReLU)
  └─ Dense(1, Linear)  →  TDEE in kcal
```

### Model 2 — Macro Recommender
```
Input (8 features incl. goal + TDEE)
  └─ Dense(128, ReLU) → BatchNorm → Dropout(0.15)
  └─ Dense(64, ReLU)  → BatchNorm → Dropout(0.10)
  └─ Dense(32, ReLU)
  └─ Dense(3, Linear)  →  [protein_g, carbs_g, fat_g]
```

### Model 3 — Food Scorer
```
Input (9 features: macros + goal index)
  └─ Dense(64, ReLU) → Dropout(0.10)
  └─ Dense(32, ReLU)
  └─ Dense(1, Sigmoid) × 100  →  score 0–100
```

---

## 📊 Data Sources

| Dataset | Source | License | Records |
|---------|--------|---------|---------|
| USDA FoodData Central | fdc.nal.usda.gov | CC0 (public domain) | 700k+ foods |
| NHANES (synthetic) | Based on CDC NHANES 2017-20 stats | Research use | 20,000 rows |
| IFCT 2017 | National Institute of Nutrition, India | Public research | 528 Indian foods |
| Open Food Facts | world.openfoodfacts.org | ODbL | 3M+ products |

> For the real NHANES microdata: https://wwwn.cdc.gov/nchs/nhanes/

---

## 🔌 API Endpoints

### `POST /api/plan`
Generate a full nutrition plan.
```json
{
  "age": 25, "sex": "male", "height_cm": 175, "weight_kg": 80,
  "body_fat_pct": 20, "activity_level": "moderate",
  "goal": "muscle", "diet_type": "none", "cuisine": "india",
  "meals_per_day": 5, "sleep_h": 7
}
```

### `GET /api/foods/search?q=paneer&goal=muscle`
Search + score foods from the database.

### `GET /api/foods/score?calories=165&protein_g=31&carbs_g=0&fat_g=3.6&goal=muscle`
Score any custom food.

### `GET /docs`
Interactive Swagger UI.

---

## 📓 Notebooks

Open `notebooks/eda_and_evaluation.ipynb` in Jupyter for:
- Plotly interactive charts (BMI vs TDEE scatter, violin plots, treemaps)
- Model evaluation metrics
- Indian food composition visualization
- Training curve gallery

```bash
pip install jupyter
jupyter notebook notebooks/eda_and_evaluation.ipynb
```

---

## 🛣 Roadmap

- [ ] Real NHANES XPT file parser
- [ ] User accounts + plan history (SQLite → PostgreSQL)
- [ ] Food photo recognition (MobileNet)  
- [ ] Meal logging + weekly adaptation
- [ ] Export plan as PDF
- [ ] React Native mobile app

---

## ⚠ Disclaimer
This tool is for educational and informational purposes only.  
Always consult a registered dietitian or physician before making significant dietary changes,  
especially if you have a medical condition.
