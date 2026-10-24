
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "diabetic_data.csv")

# Use one real patient record as test input
features = df.drop(
    columns=[
        "encounter_id", "patient_nbr", "weight",
        "payer_code", "readmitted"
    ],
    errors="ignore"
).iloc[0].to_dict()

# Convert missing values and NumPy values to JSON-safe values
features = {
    key: (
        "Unknown" if pd.isna(value) or value == "?" 
        else value.item() if hasattr(value, "item")
        else value
    )
    for key, value in features.items()
}

response = requests.post(
    "http://127.0.0.1:5000/api/predict",
    json=features
)

print("HTTP status:", response.status_code)
print("API response:", response.json())
