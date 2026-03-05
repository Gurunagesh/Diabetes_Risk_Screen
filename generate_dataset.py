"""
generate_dataset.py
====================
Generates a realistic synthetic dataset matching the structure of:
  "Diabetes Prediction in India Dataset" by ankushpanday1 (Kaggle)

Columns (20 features + 1 target):
  Age, Gender, BMI, Physical_Activity, Family_History_of_Diabetes,
  Glucose_Level, HbA1c_Level, Blood_Pressure_Systolic, Blood_Pressure_Diastolic,
  Cholesterol_Level, Smoking_Status, Alcohol_Consumption, Diet_Type,
  Stress_Level, Sleep_Duration, Waist_Circumference, Urban_Rural,
  Hypertension, Previous_Gestational_Diabetes, Medication_Use, Diabetic

REQUIRED  : Age, Gender, BMI, Physical_Activity, Family_History_of_Diabetes
OPTIONAL  : all others (imputed if missing)
TARGET    : Diabetic (0 / 1)

Replace this file's output with the real CSV when available:
  df = pd.read_csv("data/diabetes_india.csv")
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ── Demographics ──────────────────────────────────────────────────────────────
age    = np.random.randint(18, 80, N).astype(float)
gender = np.random.choice(["Male", "Female"], N, p=[0.50, 0.50])

# ── Lifestyle ─────────────────────────────────────────────────────────────────
physical_activity = np.random.choice(
    ["Sedentary", "Low", "Moderate", "High"], N, p=[0.30, 0.25, 0.30, 0.15]
)
family_history = np.random.choice(["Yes", "No"], N, p=[0.35, 0.65])
smoking        = np.random.choice(["Never", "Former", "Current"], N, p=[0.55, 0.20, 0.25])
alcohol        = np.random.choice(["None", "Occasional", "Regular"], N, p=[0.50, 0.30, 0.20])
diet_type      = np.random.choice(
    ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"], N, p=[0.40, 0.35, 0.05, 0.20]
)
urban_rural    = np.random.choice(["Urban", "Rural"], N, p=[0.55, 0.45])
stress_level   = np.random.choice(["Low", "Moderate", "High"], N, p=[0.30, 0.40, 0.30])
sleep_duration = np.round(np.random.normal(6.5, 1.2, N), 1).clip(3, 10)

# ── Anthropometric ────────────────────────────────────────────────────────────
bmi                = np.round(np.random.normal(25.5, 5.5, N), 1).clip(13, 55)
waist_circumference = np.round(
    80 + (bmi - 22) * 1.8 + np.random.normal(0, 5, N), 1
).clip(55, 140)

# ── Clinical ──────────────────────────────────────────────────────────────────
glucose   = np.round(np.random.normal(110, 32, N), 1).clip(60, 300)
hba1c     = np.round(np.random.normal(5.6, 1.1, N), 1).clip(3.5, 12.0)
bp_sys    = np.round(np.random.normal(122, 18, N), 0).clip(80, 200).astype(float)
bp_dia    = np.round(np.random.normal(78, 12, N), 0).clip(50, 120).astype(float)
chol      = np.round(np.random.normal(185, 38, N), 1).clip(100, 350)
hypert    = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
prev_gest = np.where(gender == "Female",
                     np.random.choice(["Yes", "No"], N, p=[0.12, 0.88]),
                     "No")
med_use   = np.random.choice(["Yes", "No"], N, p=[0.28, 0.72])

# ── Risk score → Diabetic label ───────────────────────────────────────────────
fh_score   = (family_history == "Yes").astype(int)
pa_score   = np.where(physical_activity == "Sedentary", 2,
             np.where(physical_activity == "Low", 1, 0))
risk = (
    (glucose > 140).astype(int)    * 3 +
    (hba1c   > 6.5).astype(int)    * 3 +
    (bmi     > 30 ).astype(int)    * 1 +
    (age     > 45 ).astype(int)    * 1 +
    fh_score                       * 1 +
    pa_score                       * 1 +
    (hypert  == "Yes").astype(int) * 1 +
    (smoking == "Current").astype(int) * 0 +
    (chol    > 240).astype(int)    * 1
)
diabetic = (risk >= 4).astype(int)
# 5% label noise
noise = np.random.random(N) < 0.05
diabetic = np.where(noise, 1 - diabetic, diabetic)

df = pd.DataFrame({
    "Age":                           age,
    "Gender":                        gender,
    "BMI":                           bmi,
    "Physical_Activity":             physical_activity,
    "Family_History_of_Diabetes":    family_history,
    "Glucose_Level":                 glucose,
    "HbA1c_Level":                   hba1c,
    "Blood_Pressure_Systolic":       bp_sys,
    "Blood_Pressure_Diastolic":      bp_dia,
    "Cholesterol_Level":             chol,
    "Smoking_Status":                smoking,
    "Alcohol_Consumption":           alcohol,
    "Diet_Type":                     diet_type,
    "Stress_Level":                  stress_level,
    "Sleep_Duration":                sleep_duration,
    "Waist_Circumference":           waist_circumference,
    "Urban_Rural":                   urban_rural,
    "Hypertension":                  hypert,
    "Previous_Gestational_Diabetes": prev_gest,
    "Medication_Use":                med_use,
    "Diabetic":                      diabetic,
})

df.to_csv("data/diabetes_india.csv", index=False)
print(f"Dataset saved → data/diabetes_india.csv")
print(f"Shape  : {df.shape}")
print(f"Diabetic prevalence : {diabetic.mean()*100:.1f}%")
print(df.head(3).to_string())

if __name__ == "__main__":
    pass   # already runs at import
