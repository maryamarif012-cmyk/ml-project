
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Project paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetic_data.csv"
MODEL_PATH = ROOT / "backend" / "models" / "model_bundle.joblib"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# Target: early readmission (<30 days)
y = (df["readmitted"] == "<30").astype(int)

# Prepare features
X = df.drop(
    columns=[
        "encounter_id", "patient_nbr", "weight",
        "payer_code", "readmitted"
    ],
    errors="ignore"
)

X = X.replace("?", "Unknown").fillna("Unknown")

age_map = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25,
    "[30-40)": 35, "[40-50)": 45, "[50-60)": 55,
    "[60-70)": 65, "[70-80)": 75, "[80-90)": 85,
    "[90-100)": 95
}
X["age"] = X["age"].map(age_map)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Preprocessing
cat_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# Train preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

positive = y_train.sum()
negative = len(y_train) - positive

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=negative / positive,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

print("Training XGBoost...")
model.fit(X_train_processed, y_train)

# Save preprocessing and model together
bundle = {
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": X.columns.tolist(),
    "age_map": age_map,
    "target_definition": "readmitted == '<30'"
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_PATH)

print("Model bundle saved successfully!")
print("Saved to:", MODEL_PATH)
print("Training complete.")
