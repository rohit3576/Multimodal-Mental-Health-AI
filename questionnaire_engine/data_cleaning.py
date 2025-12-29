import pandas as pd
import os

# =====================================================
# PATH SAFE LOAD
# =====================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

# Load dataset (DASS is tab-separated)
df = pd.read_csv(DATA_PATH, sep="\t")

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns.tolist())

# =====================================================
# STRESS QUESTIONS
# =====================================================
STRESS_COLS = ["Q1A", "Q6A", "Q8A", "Q11A", "Q12A", "Q14A", "Q18A"]

# Keep only stress-related columns
df_stress = df[STRESS_COLS].copy()

# Drop rows with missing values
df_stress.dropna(inplace=True)

# Convert to integers safely
for col in STRESS_COLS:
    df_stress[col] = pd.to_numeric(df_stress[col], errors="coerce")

df_stress.dropna(inplace=True)
df_stress = df_stress.astype(int)

print("Cleaned shape:", df_stress.shape)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df_stress["stress_score"] = df_stress.sum(axis=1)
print(df_stress["stress_score"].describe())

# =====================================================
# LABEL GENERATION
# =====================================================
def label_stress(score):
    if score >= 19:
        return "High"
    elif score >= 10:
        return "Moderate"
    else:
        return "Low"

df_stress["stress_label"] = df_stress["stress_score"].apply(label_stress)
print(df_stress["stress_label"].value_counts())

# =====================================================
# SAVE CLEAN DATASET
# =====================================================
OUTPUT_PATH = os.path.join(BASE_DIR, "stress_dataset_clean.csv")
df_stress.to_csv(OUTPUT_PATH, index=False)

print("Saved clean dataset ✔")
print("Path:", OUTPUT_PATH)
