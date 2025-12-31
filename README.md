🧠 Multimodal Mental Health AI

Multimodal Mental Health AI is an end-to-end AI system that analyzes a user’s mental health state by combining multiple signals:

✍️ Text sentiment & stress analysis (NLP)

📋 Psychological questionnaire scoring

🎥 Facial emotion analysis (video)

🧠 Explainable multimodal fusion engine

The system produces a final, confidence-aware mental health risk assessment using an interpretable fusion strategy.

⚠️ Educational & research use only — Not a medical diagnosis

🚀 Live Architecture Overview
User Input
   ├── Text (Flask UI)
   ├── Questionnaire (Flask UI)
   └── Video (Streamlit App)
            ↓
      Standardized JSON
            ↓
     Multimodal Fusion Engine
            ↓
     Final Mental Health Assessment

✨ Key Features
📝 Text Analysis (NLP)

Sentiment & stress detection

Confidence score

Explainable output

Lightweight inference pipeline

📋 Questionnaire Engine

Likert-scale mental health questionnaire

ML-based stress classification

Interpretable scoring logic

🎥 Video Emotion Engine

Facial emotion detection

Temporal smoothing

Emotion → stress mapping

Runs independently in Streamlit

🧠 Fusion Engine (Core Innovation)

Standardized schema across all modalities

Weighted multimodal fusion

Confidence estimation

Ethical medical escalation logic

🧩 Tech Stack
Backend

Python

Flask

Gunicorn

AI / ML

TensorFlow / Keras (Text model)

Scikit-learn

XGBoost (Questionnaire model)

OpenCV + FER (Video emotions)

Frontend

HTML / CSS / JavaScript

Modern glassmorphism UI

Auto-updating fusion results

Deployment

Render (Flask – free tier)

Streamlit Cloud / Local (Video engine)

📁 Project Structure
Multimodal-Mental-Health-AI/
│
├── flask_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── main.js
│
├── text_engine/
│   └── inference.py
│
├── questionnaire_engine/
│   ├── inference.py
│   ├── train_stress_model.py
│   └── data.csv
│
├── fusion_engine/
│   └── fuse_results.py
│
├── video_engine/
│   └── app.py
│
├── video_emotion/
│   └── emotion_core.py
│
├── requirements.txt
├── runtime.txt
└── README.md

🧠 Standardized Output Schema

All engines output a common JSON format:

{
  "source": "text | video | questionnaire | fusion",
  "risk_level": "Low | Moderate | High",
  "confidence": "Weak | Moderate | Strong",
  "signals": {},
  "explanation": "Human-readable explanation",
  "medical_recommendation": false
}


This enables clean fusion, transparency, and explainability.

🧪 How Fusion Works

Each modality contributes a numeric risk signal

Signals are weighted

Final risk is computed via weighted average

Confidence is derived from agreement across modalities

Ethical escalation triggers when risk is consistently high

⚙️ Local Setup
1️⃣ Clone Repository
git clone https://github.com/rohit3576/Multimodal-Mental-Health-AI.git
cd Multimodal-Mental-Health-AI

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Flask App
cd flask_app
gunicorn app:app --timeout 120

4️⃣ Run Video Engine (Separate)
cd video_engine
streamlit run app.py

☁️ Free Deployment Notes

Flask app is optimized for Render Free Tier

Heavy ML models are lazy-loaded

Training scripts are excluded from runtime

Video engine runs independently (Streamlit)

⚠️ TensorFlow models are memory-heavy — optimizations applied for free hosting.

📌 Limitations

Not a medical device

Free hosting limits model size

Video analysis requires separate service

No real-time webcam inside Flask (by design)

🛣️ Future Improvements

ONNX model conversion for lighter inference

WebSocket-based real-time updates

User authentication & history

Mobile-friendly PWA version

Cloud-based video inference API

👤 Author

Rohit Pawar
AI / ML • MERN Stack • Full-Stack Developer

🔗 GitHub: https://github.com/rohit3576

⚠️ Disclaimer

This project is for educational and research purposes only.
It does not replace professional mental health advice or diagnosis.
