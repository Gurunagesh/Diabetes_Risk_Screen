"""
utils/predictor.py
Core prediction engine:
  - Validates required inputs
  - Encodes categoricals
  - KNN-imputes missing optional fields
  - Scales features
  - Runs all trained models
  - Returns ensemble result + imputed field log
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── Feature definitions ───────────────────────────────────────────────────────
REQUIRED_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Physical_Activity",
    "Family_History_of_Diabetes",
]

OPTIONAL_FEATURES = [
    "Glucose_Level",
    "HbA1c_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Cholesterol_Level",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Diet_Type",
    "Stress_Level",
    "Sleep_Duration",
    "Waist_Circumference",
    "Urban_Rural",
    "Hypertension",
    "Previous_Gestational_Diabetes",
    "Medication_Use",
]

ALL_FEATURES = REQUIRED_FEATURES + OPTIONAL_FEATURES

CATEGORICAL_COLS = [
    "Gender",
    "Physical_Activity",
    "Family_History_of_Diabetes",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Diet_Type",
    "Stress_Level",
    "Urban_Rural",
    "Hypertension",
    "Previous_Gestational_Diabetes",
    "Medication_Use",
]

# Human-readable labels for the UI
FEATURE_LABELS = {
    "Age":                           "Age (years)",
    "Gender":                        "Gender",
    "BMI":                           "BMI (kg/m²)",
    "Physical_Activity":             "Physical Activity Level",
    "Family_History_of_Diabetes":    "Family History of Diabetes",
    "Glucose_Level":                 "Blood Glucose Level (mg/dL)",
    "HbA1c_Level":                   "HbA1c Level (%)",
    "Blood_Pressure_Systolic":       "Systolic Blood Pressure (mmHg)",
    "Blood_Pressure_Diastolic":      "Diastolic Blood Pressure (mmHg)",
    "Cholesterol_Level":             "Cholesterol Level (mg/dL)",
    "Smoking_Status":                "Smoking Status",
    "Alcohol_Consumption":           "Alcohol Consumption",
    "Diet_Type":                     "Diet Type",
    "Stress_Level":                  "Stress Level",
    "Sleep_Duration":                "Sleep Duration (hours/day)",
    "Waist_Circumference":           "Waist Circumference (cm)",
    "Urban_Rural":                   "Residence Type",
    "Hypertension":                  "Hypertension",
    "Previous_Gestational_Diabetes": "Previous Gestational Diabetes",
    "Medication_Use":                "Currently on Diabetes Medication",
}

NORMAL_RANGES = {
    "Glucose_Level":            "Normal <100 · Pre-diabetic 100-125 · Diabetic ≥126 mg/dL",
    "HbA1c_Level":              "Normal <5.7% · Pre-diabetic 5.7-6.4% · Diabetic ≥6.5%",
    "BMI":                      "Underweight <18.5 · Normal 18.5-24.9 · Overweight 25-29.9 · Obese ≥30",
    "Blood_Pressure_Systolic":  "Normal <120 · Elevated 120-129 · High ≥130 mmHg",
    "Blood_Pressure_Diastolic": "Normal <80 · High ≥80 mmHg",
    "Cholesterol_Level":        "Desirable <200 · Borderline 200-239 · High ≥240 mg/dL",
    "Sleep_Duration":           "Recommended 7-9 hours/day",
    "Waist_Circumference":      "Low risk Male <94cm Female <80cm",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _load_all():
    """Load imputer, encoders, scaler and all trained models."""
    required_files = ["knn_imputer.pkl", "scaler.pkl",
                      "label_encoders.pkl", "meta.pkl"]
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Please run  python train_models.py  first."
        )

    imputer  = joblib.load(os.path.join(MODELS_DIR, "knn_imputer.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))

    model_files = {
        "Random Forest":       "rf_model.pkl",
        "XGBoost":             "xgb_model.pkl",
        "SVM":                 "svm_model.pkl",
        "Logistic Regression": "lr_model.pkl",
        "Gradient Boosting":   "gb_model.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)

    dl_path = os.path.join(MODELS_DIR, "dl_model.h5")
    if os.path.exists(dl_path):
        try:
            from tensorflow.keras.models import load_model
            models["Deep Learning ANN"] = load_model(dl_path)
        except Exception:
            pass

    return imputer, scaler, encoders, models


def encode_input(raw: dict, encoders: dict) -> dict:
    """Encode categorical values; return NaN for missing."""
    encoded = {}
    for feat in ALL_FEATURES:
        val = raw.get(feat)
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            encoded[feat] = np.nan
        elif feat in CATEGORICAL_COLS and feat in encoders:
            le = encoders[feat]
            str_val = str(val)
            if str_val in le.classes_:
                encoded[feat] = float(le.transform([str_val])[0])
            else:
                # Unseen category — use mode (0)
                encoded[feat] = 0.0
        else:
            try:
                encoded[feat] = float(val)
            except (ValueError, TypeError):
                encoded[feat] = np.nan
    return encoded


def predict(user_inputs: dict) -> dict:
    """
    Run full prediction pipeline.

    Parameters
    ----------
    user_inputs : dict
        Keys from ALL_FEATURES.
        REQUIRED keys must be provided (non-None, non-empty).
        OPTIONAL keys can be None → auto-imputed via KNN.

    Returns
    -------
    dict:
        models               – per-model {prediction, probability}
        ensemble_prediction  – 0 or 1
        ensemble_probability – avg probability across models (%)
        imputed_fields       – {feature: imputed_value} for auto-filled fields
        processed_values     – all feature values after imputation (decoded)
        risk_level           – "Low" / "Moderate" / "High"
    """
    # ── Validate required ─────────────────────────────────────────────────────
    missing_req = []
    for feat in REQUIRED_FEATURES:
        val = user_inputs.get(feat)
        if val is None or val == "":
            missing_req.append(FEATURE_LABELS.get(feat, feat))
    if missing_req:
        raise ValueError(f"Required fields missing: {', '.join(missing_req)}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    imputer, scaler, encoders, models = _load_all()

    # ── Track which optional fields were omitted ──────────────────────────────
    omitted = [f for f in OPTIONAL_FEATURES
               if user_inputs.get(f) is None or user_inputs.get(f) == ""]

    # ── Encode ────────────────────────────────────────────────────────────────
    encoded = encode_input(user_inputs, encoders)
    row = [encoded[f] for f in ALL_FEATURES]
    df_in = pd.DataFrame([row], columns=ALL_FEATURES)

    # ── Impute ────────────────────────────────────────────────────────────────
    df_imp = pd.DataFrame(imputer.transform(df_in), columns=ALL_FEATURES)

    # ── Scale ─────────────────────────────────────────────────────────────────
    X = scaler.transform(df_imp)

    # ── Predict ───────────────────────────────────────────────────────────────
    results = {}
    for name, model in models.items():
        if name == "Deep Learning ANN":
            prob = float(model.predict(X, verbose=0)[0][0])
        else:
            prob = float(model.predict_proba(X)[0][1])
        results[name] = {
            "prediction":  int(prob >= 0.5),
            "probability": round(prob * 100, 1),
        }

    # ── Ensemble ──────────────────────────────────────────────────────────────
    avg_prob   = float(np.mean([v["probability"] for v in results.values()]))
    ens_pred   = int(avg_prob >= 50)
    risk_level = "Low" if avg_prob < 35 else ("Moderate" if avg_prob < 65 else "High")

    # ── Decode imputed values for display ─────────────────────────────────────
    imputed_display = {}
    for feat in omitted:
        raw_val = df_imp[feat].iloc[0]
        if feat in CATEGORICAL_COLS and feat in encoders:
            le = encoders[feat]
            idx = int(round(raw_val))
            idx = max(0, min(idx, len(le.classes_) - 1))
            imputed_display[feat] = le.classes_[idx]
        else:
            imputed_display[feat] = round(raw_val, 2)

    return {
        "models":               results,
        "ensemble_prediction":  ens_pred,
        "ensemble_probability": round(avg_prob, 1),
        "imputed_fields":       imputed_display,
        "risk_level":           risk_level,
    }
