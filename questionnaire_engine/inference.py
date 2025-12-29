import os
import joblib
import numpy as np
import pandas as pd

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "questionnaire", "stress_model.pkl"
)

# =====================================================
# LOAD MODEL BUNDLE
# =====================================================
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
label_mapping = bundle["label_mapping"]

# =====================================================
# INFERENCE FUNCTION (STANDARDIZED OUTPUT)
# =====================================================
def analyze_questionnaire(answers: dict):
    """
    Perform stress analysis using questionnaire responses.

    Returns standardized JSON-safe dict
    compatible with fusion & UI layers.
    """

    # Convert answers dict → DataFrame
    df = pd.DataFrame([answers])

    # Ensure correct feature order
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Model prediction
    pred_idx = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]

    risk_level = label_mapping[pred_idx]
    confidence = float(np.max(proba))

    # Simple explainable stress score
    stress_score = int(sum(answers.values()))

    return {
        "source": "questionnaire",
        "risk_level": risk_level,
        "confidence": round(confidence, 2),
        "signals": {
            "stress_score": stress_score,
            "answers_used": len(answers)
        },
        "explanation": "Questionnaire responses indicate stress patterns."
    }
