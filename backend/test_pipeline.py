
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

bundle = joblib.load(ROOT / "backend" / "models" / "model_bundle.joblib")

df = pd.read_csv(ROOT / "data" / "diabetic_data.csv")

# Use one real record as a test input
X = df.drop(
    columns=[
        "encounter_id", "patient_nbr", "weight",
        "payer_code", "readmitted"
    ],
    errors="ignore"
).iloc[[0]].copy()

X = X.replace("?", "Unknown").fillna("Unknown")
X["age"] = X["age"].map(bundle["age_map"])

# Apply saved preprocessing and model
X_processed = bundle["preprocessor"].transform(X)
probability = bundle["model"].predict_proba(X_processed)[0, 1]
prediction = int(probability >= 0.5)

print("Saved model loaded successfully.")
print("Early readmission probability:", round(float(probability), 4))
print("Predicted class:", prediction)
print("0 = Not predicted as early readmission")
print("1 = Predicted as early readmission")
