
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_model_pipeline():
    # Load the saved model bundle
    bundle = joblib.load(
        ROOT / "backend" / "models" / "model_bundle.joblib"
    )

    # Load the dataset
    df = pd.read_csv(ROOT / "data" / "diabetic_data.csv")

    # Prepare one real record
    X = df.drop(
        columns=[
            "encounter_id",
            "patient_nbr",
            "weight",
            "payer_code",
            "readmitted",
        ],
        errors="ignore",
    ).iloc[[0]].copy()

    X = X.replace("?", "Unknown").fillna("Unknown")
    X["age"] = X["age"].map(bundle["age_map"])

    # Apply preprocessing and make a prediction
    X_processed = bundle["preprocessor"].transform(X)
    probability = bundle["model"].predict_proba(X_processed)[0, 1]
    prediction = int(probability >= 0.5)

    # Verify the prediction is valid
    assert 0.0 <= float(probability) <= 1.0
    assert prediction in (0, 1)

    print("Saved model loaded successfully.")
    print("Early readmission probability:", round(float(probability), 4))
    print("Predicted class:", prediction)
