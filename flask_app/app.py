from flask import Flask, render_template, request, jsonify
import os
import sys

# =====================================================
# PATH FIX (PROJECT ROOT)
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# IMPORT AI ENGINES
# =====================================================
from text_engine.inference import analyze_text
from questionnaire_engine.inference import analyze_questionnaire
from fusion_engine.fuse_results import fuse_results

# =====================================================
# APP INIT
# =====================================================
app = Flask(__name__)

# =====================================================
# GLOBAL STATE (SAFE, SIMPLE)
# =====================================================
text_result = None
questionnaire_result = None
video_result = None        # comes from Streamlit
fusion_result = None

# =====================================================
# API: RECEIVE VIDEO RESULT FROM STREAMLIT
# =====================================================
@app.route("/api/video-result", methods=["POST"])
def receive_video_result():
    global video_result, fusion_result

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "No JSON received"
        }), 400

    video_result = data

    print("\n📥 Video Result Received:")
    print(video_result)

    # Re-run fusion when video arrives
    fusion_result = fuse_results(
        text_result=text_result,
        questionnaire_result=questionnaire_result,
        video_result=video_result
    )

    return jsonify({
        "status": "success",
        "message": "Video result stored & fused"
    }), 200


# =====================================================
# OPTIONAL: STATUS API (for auto-refresh later)
# =====================================================
@app.route("/api/status")
def api_status():
    return jsonify({
        "text": text_result is not None,
        "questionnaire": questionnaire_result is not None,
        "video": video_result is not None,
        "fusion": fusion_result is not None
    })


# =====================================================
# MAIN UI ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    global text_result, questionnaire_result, fusion_result

    if request.method == "POST":

        # ================= TEXT ANALYSIS =================
        if "text" in request.form and request.form["text"].strip():
            text_result = analyze_text(request.form["text"])

        # ================= QUESTIONNAIRE =================
        questionnaire_keys = [k for k in request.form if k.startswith("Q")]

        if questionnaire_keys:
            questionnaire_answers = {
                k: int(request.form[k])
                for k in questionnaire_keys
                if request.form[k].isdigit()
            }

            if questionnaire_answers:
                questionnaire_result = analyze_questionnaire(
                    questionnaire_answers
                )

        # ================= FUSION =================
        fusion_result = fuse_results(
            text_result=text_result,
            questionnaire_result=questionnaire_result,
            video_result=video_result
        )

    return render_template(
        "index.html",
        text_result=text_result,
        questionnaire_result=questionnaire_result,
        video_result=video_result,
        fusion_result=fusion_result
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False   # important on Windows
    )
