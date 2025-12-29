import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# =====================================================
# PATHS (MATCH CURRENT PROJECT STRUCTURE)
# =====================================================
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sentiment")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "preprocess_config.pkl")

# =====================================================
# LOAD MODEL + CONFIG
# =====================================================
sentiment_model = tf.keras.models.load_model(MODEL_PATH)

with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)

VOCAB_SIZE = config["vocab_size"]
MAX_LEN = config["max_len"]

# Load IMDB word index once
word_index = imdb.get_word_index()

# =====================================================
# TEXT ENCODING
# =====================================================
def encode_text(text: str):
    words = text.lower().split()
    encoded = []

    for word in words:
        idx = word_index.get(word, 2)  # 2 = OOV token
        encoded.append(idx if idx < VOCAB_SIZE else 2)

    if not encoded:
        encoded = [2]

    return pad_sequences(
        [encoded],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

# =====================================================
# INFERENCE FUNCTION (STANDARDIZED OUTPUT)
# =====================================================
def analyze_text(text: str):
    """
    Perform sentiment + stress inference on input text.

    Returns standardized JSON-safe dict
    compatible with fusion & UI layers.
    """
    encoded = encode_text(text)

    score = float(
        sentiment_model.predict(encoded, verbose=0)[0][0]
    )

    sentiment = "Positive" if score >= 0.5 else "Negative"

    # Explainable risk mapping
    if score >= 0.65:
        risk_level = "Low"
    elif score >= 0.45:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "source": "text",
        "risk_level": risk_level,
        "confidence": round(score, 2),
        "signals": {
            "sentiment": sentiment,
            "sentiment_score": round(score, 2)
        },
        "explanation": "Text sentiment analysis indicates emotional stress level."
    }
