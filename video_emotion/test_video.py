import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from video_emotion.emotion_detector import analyze_video

if __name__ == "__main__":
    print("🎥 Starting facial emotion analysis...")
    result = analyze_video(0)
    print("\n🧠 Analysis Result:")
    print(result)
