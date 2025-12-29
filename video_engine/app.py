# =========================================================
# PATH FIX — MUST BE AT THE VERY TOP
# =========================================================
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import cv2
import time
import tempfile
import requests
from fer import FER
from collections import deque, Counter
from video_emotion.emotion_core import analyze_frames

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Video Emotion Engine",
    page_icon="🎥",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.25);
}
.center { text-align: center; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown("<h1 class='center'>🎥 AI Video Emotion Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center'>Stable real-time facial emotion detection (free-deploy ready)</p>",
    unsafe_allow_html=True
)

# =========================================================
# INPUT MODE
# =========================================================
mode = st.radio(
    "Choose Input Mode:",
    ["🎥 Webcam", "📁 Upload Video"],
    horizontal=True
)

start = st.button("🚀 Start Analysis", use_container_width=True)

# =========================================================
# FACE DETECTOR (STABLE MODE)
# =========================================================
# ❌ mtcnn=True causes Conv2D crashes on empty faces
# ✅ Haar cascade is stable for free deployment
face_detector = FER(mtcnn=False)

# =========================================================
# SMOOTHING CONFIG
# =========================================================
SMOOTHING_WINDOW = 7
emotion_buffer = deque(maxlen=SMOOTHING_WINDOW)
confidence_buffer = deque(maxlen=SMOOTHING_WINDOW)

NEGATIVE = {"angry", "sad", "fear", "disgust"}

# =========================================================
# RISK MAP
# =========================================================
def emotion_to_risk(emotion):
    if emotion in NEGATIVE:
        return "High"
    elif emotion == "neutral":
        return "Moderate"
    return "Low"

# =========================================================
# SMOOTHING
# =========================================================
def smooth_emotion(emotion, confidence):
    emotion_buffer.append(emotion)
    confidence_buffer.append(confidence)

    dominant = Counter(emotion_buffer).most_common(1)[0][0]
    avg_conf = sum(confidence_buffer) / len(confidence_buffer)

    return dominant, avg_conf

# =========================================================
# SAFE WEBCAM STREAM
# =========================================================
def webcam_stream(preview, emotion_counter):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    last_capture = time.time()

    while time.time() - start_time <= 15:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        if h < 100 or w < 100:
            continue

        if time.time() - last_capture < 0.3:
            continue
        last_capture = time.time()

        try:
            detections = face_detector.detect_emotions(frame)
            if not detections:
                preview.image(frame, channels="BGR")
                continue
        except Exception:
            preview.image(frame, channels="BGR")
            continue

        for d in detections:
            emotions = d.get("emotions", {})
            if not emotions:
                continue

            emo = max(emotions, key=emotions.get)
            conf = emotions[emo]

            emotion_counter[emo] += 1
            smooth_label, smooth_conf = smooth_emotion(emo, conf)

            x, y, w, h = d["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{smooth_label} ({int(smooth_conf*100)}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        preview.image(frame, channels="BGR")
        yield frame

    cap.release()

# =========================================================
# SEND TO FLASK
# =========================================================
def send_to_flask(payload):
    FLASK_API = "http://127.0.0.1:5000/api/video-result"
    try:
        r = requests.post(FLASK_API, json=payload, timeout=5)
        if r.status_code == 200:
            st.success("📤 Result sent to Flask")
        else:
            st.error("❌ Flask rejected payload")
    except Exception:
        st.error("❌ Flask server not running")

# =========================================================
# MAIN EXECUTION
# =========================================================
if start:
    # RESET STATE
    emotion_buffer.clear()
    confidence_buffer.clear()
    emotion_counter = Counter()

    preview = st.empty()

    with st.spinner("Analyzing facial emotions (~15s)…"):
        if mode == "🎥 Webcam":
            result = analyze_frames(webcam_stream(preview, emotion_counter))
        else:
            uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
            if uploaded is None:
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            result = analyze_frames(webcam_stream(preview, emotion_counter))

    # SAFE FALLBACK
    dominant_emotion = (
        emotion_counter.most_common(1)[0][0]
        if emotion_counter else "neutral"
    )

    total = sum(emotion_counter.values()) or 1

    video_payload = {
        "source": "video",
        "risk_level": emotion_to_risk(dominant_emotion),
        "confidence": round(result.get("reliability", 0.7), 2),
        "signals": {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": {
                k: round((v / total) * 100, 1)
                for k, v in emotion_counter.items()
            }
        },
        "explanation": "Facial emotion analysis over 15 seconds"
    }

    st.success("✅ Analysis Complete")
    st.markdown("### 🧠 Standardized Video Output")
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.json(video_payload)
    st.markdown("</div>", unsafe_allow_html=True)

    send_to_flask(video_payload)
