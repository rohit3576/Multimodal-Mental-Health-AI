import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "stress_dataset_clean.csv"))

X = df.drop(columns=["stress_score", "stress_label"])
y = df["stress_label"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,        # like epochs
    learning_rate=0.05,
    max_depth=4,
    eval_metric="mlogloss"
)

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=True
)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump(
    {"model": model, "label_encoder": le, "features": X.columns.tolist()},
    os.path.join(BASE_DIR, "stress_model_xgb.pkl")
)

# Plot training curves
results = model.evals_result()

plt.plot(results["validation_0"]["mlogloss"], label="Train")
plt.plot(results["validation_1"]["mlogloss"], label="Validation")
plt.legend()
plt.title("XGBoost Training Curve")
plt.xlabel("Boosting Rounds")
plt.ylabel("Log Loss")
plt.show()
