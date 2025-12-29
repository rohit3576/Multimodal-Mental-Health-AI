import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "stress_dataset_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", "questionnaire")
MODEL_PATH = os.path.join(MODEL_DIR, "stress_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["stress_label"])
y = df["stress_label"]

# =====================================================
# ENCODE LABELS
# =====================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =====================================================
# SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================================
# MODEL
# =====================================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

print("\n📊 Classification Report\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n🧩 Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# SAVE BUNDLE (IMPORTANT)
# =====================================================
bundle = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "label_mapping": label_encoder.classes_.tolist()
}

joblib.dump(bundle, MODEL_PATH)

print(f"\n✅ Model bundle saved at: {MODEL_PATH}")
