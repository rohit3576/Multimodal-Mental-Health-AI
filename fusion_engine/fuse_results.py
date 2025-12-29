from typing import Dict, Optional, List

# =====================================================
# RISK NORMALIZATION
# =====================================================
RISK_MAP = {
    "Low": 1,
    "Moderate": 2,
    "High": 3
}

# =====================================================
# MODALITY WEIGHTS (IMPORTANT)
# =====================================================
WEIGHTS = {
    "text": 0.2,
    "video": 0.3,
    "questionnaire": 0.5
}

# =====================================================
# FUSION FUNCTION (WEIGHTED + STANDARDIZED)
# =====================================================
def fuse_results(
    text_result: Optional[Dict] = None,
    video_result: Optional[Dict] = None,
    questionnaire_result: Optional[Dict] = None
) -> Dict:
    """
    Weighted multimodal fusion for mental health risk assessment.

    Expected input schema (each engine):
    {
        "source": "text | video | questionnaire",
        "risk_level": "Low | Moderate | High",
        "confidence": float (0–1),
        "signals": {...},
        "explanation": str
    }
    """

    weighted_scores: List[float] = []
    source_risks: Dict[str, str] = {}
    confidences: List[float] = []

    # =====================================================
    # INGEST FUNCTION
    # =====================================================
    def ingest(result: Optional[Dict]):
        if not result:
            return

        source = result.get("source")
        risk = result.get("risk_level")
        confidence = float(result.get("confidence", 0.7))

        if source not in WEIGHTS:
            return
        if risk not in RISK_MAP:
            return

        weight = WEIGHTS[source]
        numeric_risk = RISK_MAP[risk]

        weighted_scores.append(numeric_risk * weight)
        source_risks[source] = risk
        confidences.append(confidence)

    # =====================================================
    # INGEST ALL SOURCES
    # =====================================================
    ingest(text_result)
    ingest(video_result)
    ingest(questionnaire_result)

    # =====================================================
    # SAFETY CHECK
    # =====================================================
    if not weighted_scores:
        return {
            "source": "fusion",
            "risk_level": "Unknown",
            "confidence": {
                "label": "Weak",
                "score": 0.0
            },
            "signals": {},
            "explanation": "Insufficient multimodal data for assessment.",
            "medical_recommendation": False
        }

    # =====================================================
    # WEIGHTED FUSION LOGIC
    # =====================================================
    weighted_avg = round(sum(weighted_scores) / sum(
        WEIGHTS[s] for s in source_risks
    ), 2)

    if weighted_avg >= 2.5:
        final_risk = "High"
    elif weighted_avg >= 1.7:
        final_risk = "Moderate"
    else:
        final_risk = "Low"

    # =====================================================
    # CONFIDENCE LOGIC
    # =====================================================
    agreement_range = max(RISK_MAP[r] for r in source_risks.values()) - \
                      min(RISK_MAP[r] for r in source_risks.values())

    avg_confidence = round(sum(confidences) / len(confidences), 2)

    if agreement_range == 0:
        confidence_label = "Strong"
    elif agreement_range == 1:
        confidence_label = "Moderate"
    else:
        confidence_label = "Weak"

    # =====================================================
    # MEDICAL ESCALATION (ETHICAL)
    # =====================================================
    high_sources = sum(
        1 for r in source_risks.values() if r == "High"
    )

    medical_recommendation = (
        high_sources >= 2 or
        (final_risk == "High" and "questionnaire" in source_risks)
    )

    # =====================================================
    # FINAL OUTPUT (STANDARDIZED)
    # =====================================================
    return {
        "source": "fusion",
        "risk_level": final_risk,
        "confidence": {
            "label": confidence_label,
            "score": avg_confidence
        },
        "signals": source_risks,
        "explanation": (
            "Weighted fusion applied: questionnaire (0.5), "
            "video (0.3), text (0.2)."
        ),
        "medical_recommendation": medical_recommendation
    }
