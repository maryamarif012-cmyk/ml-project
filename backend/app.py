
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "backend" / "models" / "model_bundle.joblib"

bundle = joblib.load(MODEL_PATH)
preprocessor = bundle["preprocessor"]
model = bundle["model"]
feature_columns = bundle["feature_columns"]
age_map = bundle["age_map"]

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/api/sample", methods=["GET"])
def sample():
    df = pd.read_csv(ROOT / "data" / "diabetic_data.csv")

    features = df.drop(
        columns=[
            "encounter_id", "patient_nbr", "weight",
            "payer_code", "readmitted"
        ],
        errors="ignore"
    ).iloc[0].to_dict()

    features = {
        key: (
            "Unknown" if pd.isna(value) or value == "?"
            else value.item() if hasattr(value, "item")
            else value
        )
        for key, value in features.items()
    }

    return jsonify(features)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send patient data as a JSON object"}), 400

    missing = [col for col in feature_columns if col not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        X = pd.DataFrame([data], columns=feature_columns)
        X = X.replace("?", "Unknown").fillna("Unknown")
        X["age"] = X["age"].map(age_map)

        if X["age"].isna().any():
            return jsonify({"error": "Invalid age category"}), 400

        X_processed = preprocessor.transform(X)
        probability = float(model.predict_proba(X_processed)[0, 1])
        prediction = int(probability >= 0.5)

        return jsonify({
            "predicted_class": prediction,
            "early_readmission_probability": round(probability, 4),
            "interpretation": (
                "Predicted early readmission"
                if prediction == 1
                else "Not predicted as early readmission"
            ),
            "notice": "Experimental model output; not a clinical diagnosis."
        })

    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
