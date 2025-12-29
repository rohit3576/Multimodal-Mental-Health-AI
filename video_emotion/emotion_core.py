import time
import numpy as np
from fer import FER
from collections import defaultdict

# ================= CONFIG =================
ANALYSIS_SECONDS = 15
CONFIDENCE_THRESHOLD = 0.5
MIN_EMOTION_SAMPLES = 20

NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}
POSITIVE_EMOTIONS = {"happy", "surprise"}

# 🚨 IMPORTANT FIX:
# Core logic MUST NOT use MTCNN
# Face detection already happens in UI layer
detector = FER(mtcnn=False)

# ================= CORE FUNCTION =================
def analyze_frames(frame_generator):
    """
    Analyze emotions from a stream of frames.
    Frame source is controlled externally (Streamlit).
    """

    emotion_scores = defaultdict(list)
    total_frames = 0
    valid_frames = 0

    start_time = time.time()

    for frame in frame_generator:

        if time.time() - start_time >= ANALYSIS_SECONDS:
            break

        total_frames += 1

        try:
            detections = detector.detect_emotions(frame)
        except Exception:
            continue

        if not detections:
            continue

        valid_frames += 1

        # FER without MTCNN returns at most one face
        emotions = detections[0].get("emotions", {})

        for emotion, score in emotions.items():
            if score >= CONFIDENCE_THRESHOLD:
                emotion_scores[emotion].append(score)

        # Early exit if enough emotion evidence collected
        if sum(len(v) for v in emotion_scores.values()) >= MIN_EMOTION_SAMPLES:
            break

    # ================= NO FACE CASE =================
    if not emotion_scores:
        return {
            "status": "no_face_detected",
            "message": "Face not detected clearly.",
            "reliability": 0.0
        }

    # ================= AGGREGATION =================
    averaged = {
        emotion: float(np.mean(scores))
        for emotion, scores in emotion_scores.items()
    }

    dominant_emotion = max(averaged, key=averaged.get)
    total_score = sum(averaged.values())

    emotion_distribution = {
        e: round((s / total_score) * 100, 2)
        for e, s in averaged.items()
    }

    # ================= PSYCHOLOGICAL LOGIC =================
    negative_score = sum(
        s for e, s in averaged.items() if e in NEGATIVE_EMOTIONS
    )
    positive_score = sum(
        s for e, s in averaged.items() if e in POSITIVE_EMOTIONS
    )

    margin = abs(negative_score - positive_score)

    if margin < 0.15:
        emotional_state = "Uncertain"
        stress_risk = "Moderate"
    elif negative_score > positive_score:
        emotional_state = "Negative"
        stress_risk = "High"
    else:
        emotional_state = "Positive"
        stress_risk = "Low"

    # ================= RELIABILITY =================
    reliability = round(
        min(1.0, valid_frames / max(1, total_frames)),
        2
    )

    return {
        "status": "success",
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "emotional_state": emotional_state,
        "stress_risk": stress_risk,
        "analysis_seconds": round(time.time() - start_time, 2),
        "reliability": reliability
    }
