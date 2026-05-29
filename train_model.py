import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 1. Generate dataset if it doesn't exist ──────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

DATA_PATH = os.path.join("data", "insurance.csv")

if not os.path.exists(DATA_PATH):
    print("Generating dataset ...")
    np.random.seed(42)
    n = 1000
    age      = np.random.randint(18, 65, n)
    bmi      = np.round(np.random.uniform(18.0, 45.0, n), 1)
    bp       = np.random.randint(60, 130, n)
    children = np.random.randint(0, 5, n)
    smoker   = np.random.choice([0, 1], n, p=[0.8, 0.2])
    region   = np.random.choice(["northeast","northwest","southeast","southwest"], n)
    sex      = np.random.choice(["male","female"], n)
    charges  = np.round(np.clip(
        age*250 + bmi*120 + bp*80 + children*500
        + smoker*15000 + np.random.normal(0, 1500, n),
        1000, 70000
    ), 2)
    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi, "blood_pressure": bp,
        "children": children, "smoker": smoker, "region": region, "charges": charges
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"  Saved {len(df)} rows → {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded existing dataset: {df.shape}")

# ── 2. Encode ─────────────────────────────────────────────────────────────────
le_sex    = LabelEncoder()
le_region = LabelEncoder()

df["sex_enc"]    = le_sex.fit_transform(df["sex"])
df["region_enc"] = le_region.fit_transform(df["region"])

FEATURE_COLS = ["age","bmi","blood_pressure","children","smoker","sex_enc","region_enc"]
X = df[FEATURE_COLS]
y = df["charges"]

# ── 3. Split & scale ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ── 4. Train AdaBoost ─────────────────────────────────────────────────────────
print("\nTraining AdaBoost Regressor ...")
model = AdaBoostRegressor(
    estimator    = DecisionTreeRegressor(max_depth=4),
    n_estimators = 100,
    learning_rate= 0.1,
    random_state = 42,
)
model.fit(X_train_sc, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"\n  R²   : {r2:.4f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  MAE  : ${mae:,.2f}")

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model,     os.path.join("models", "adaboost_model.pkl"))
joblib.dump(scaler,    os.path.join("models", "scaler.pkl"))
joblib.dump(le_sex,    os.path.join("models", "le_sex.pkl"))
joblib.dump(le_region, os.path.join("models", "le_region.pkl"))

print("\nAll artifacts saved to models/")
print("adaboost_model.pkl")
print("scaler.pkl")
print("le_sex.pkl")
print("le_region.pkl")