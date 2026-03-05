"""
train_models.py  —  Run once before launching the app:  python train_models.py
Place real Kaggle CSV at data/diabetes_india.csv (auto-generates synthetic if absent).
"""
import os, sys, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.impute           import KNNImputer
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix)

try:
    from imblearn.over_sampling import SMOTE; HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False; print("[INFO] No imbalanced-learn → skipping SMOTE")
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("[INFO] No xgboost → skipping XGBoost")
try:
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False; print("[INFO] No TensorFlow → skipping Deep Learning")

print("="*60); print("  Smart Diabetes Detection — Training"); print("="*60)

REQUIRED_FEATURES = ["Age","Gender","BMI","Physical_Activity","Family_History_of_Diabetes"]
OPTIONAL_FEATURES = ["Glucose_Level","HbA1c_Level","Blood_Pressure_Systolic",
                     "Blood_Pressure_Diastolic","Cholesterol_Level","Smoking_Status",
                     "Alcohol_Consumption","Diet_Type","Stress_Level","Sleep_Duration",
                     "Waist_Circumference","Urban_Rural","Hypertension",
                     "Previous_Gestational_Diabetes","Medication_Use"]
ALL_FEATURES  = REQUIRED_FEATURES + OPTIONAL_FEATURES
TARGET        = "Diabetic"
CATEGORICAL_COLS = ["Gender","Physical_Activity","Family_History_of_Diabetes",
                    "Smoking_Status","Alcohol_Consumption","Diet_Type","Stress_Level",
                    "Urban_Rural","Hypertension","Previous_Gestational_Diabetes","Medication_Use"]
NUMERIC_COLS  = [c for c in ALL_FEATURES if c not in CATEGORICAL_COLS]

# ── Load / generate data ──────────────────────────────────────────────────────
CSV = "data/diabetes_india.csv"
if os.path.exists(CSV):
    df = pd.read_csv(CSV); print(f"\nLoaded CSV: {df.shape}")
else:
    print("\nGenerating synthetic dataset…")
    import generate_dataset; df = pd.read_csv(CSV)
    print(f"Generated: {df.shape}")
print(f"Diabetic prevalence: {df[TARGET].mean()*100:.1f}%")

# ── Encode ────────────────────────────────────────────────────────────────────
print("\nEncoding categoricals…")
label_encoders = {}; df_enc = df[ALL_FEATURES+[TARGET]].copy()
for col in CATEGORICAL_COLS:
    if col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str).fillna("Unknown"))
        label_encoders[col] = le
joblib.dump(label_encoders, "models/label_encoders.pkl")
print(f"  Encoded {len(label_encoders)} columns.")

# ── Impute ────────────────────────────────────────────────────────────────────
print("Fitting KNN Imputer (k=5)…")
X_raw = df_enc[ALL_FEATURES].values.astype(float)
y     = df_enc[TARGET].values.astype(int)
imputer = KNNImputer(n_neighbors=5)
X_imp   = imputer.fit_transform(pd.DataFrame(X_raw, columns=ALL_FEATURES))
joblib.dump(imputer, "models/knn_imputer.pkl")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(pd.DataFrame(X_imp, columns=ALL_FEATURES))
joblib.dump(scaler, "models/scaler.pkl")

# ── Balance + split ───────────────────────────────────────────────────────────
if HAS_SMOTE:
    smote = SMOTE(random_state=42); X_sc, y = smote.fit_resample(X_sc, y)
    print("SMOTE applied.")
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=.2, random_state=42, stratify=y)
print(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

# ── Train ─────────────────────────────────────────────────────────────────────
metrics = {}
def ev(name, model, dl_prob=None):
    pred = model.predict(X_te) if dl_prob is None else (dl_prob>=.5).astype(int)
    prob = model.predict_proba(X_te)[:,1] if dl_prob is None else dl_prob
    m = {"accuracy":round(accuracy_score(y_te,pred)*100,2),
         "precision":round(precision_score(y_te,pred)*100,2),
         "recall":round(recall_score(y_te,pred)*100,2),
         "f1":round(f1_score(y_te,pred)*100,2),
         "roc_auc":round(roc_auc_score(y_te,prob)*100,2),
         "cm":confusion_matrix(y_te,pred).tolist()}
    print(f"  {name:28s} Acc={m['accuracy']}% F1={m['f1']}% AUC={m['roc_auc']}%")
    return m

print("\nTraining models…")
rf = RandomForestClassifier(200, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_tr,y_tr); joblib.dump(rf,"models/rf_model.pkl"); metrics["Random Forest"]=ev("Random Forest",rf)

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
gb.fit(X_tr,y_tr); joblib.dump(gb,"models/gb_model.pkl"); metrics["Gradient Boosting"]=ev("Gradient Boosting",gb)

svm = SVC(kernel="rbf",C=1.0,probability=True,random_state=42)
svm.fit(X_tr,y_tr); joblib.dump(svm,"models/svm_model.pkl"); metrics["SVM"]=ev("SVM",svm)

lr = LogisticRegression(C=1.0,max_iter=1000,random_state=42)
lr.fit(X_tr,y_tr); joblib.dump(lr,"models/lr_model.pkl"); metrics["Logistic Regression"]=ev("Logistic Regression",lr)

if HAS_XGB:
    xm = xgb.XGBClassifier(200,.05,max_depth=6,subsample=.8,colsample_bytree=.8,
                            eval_metric="logloss",random_state=42)
    xm.fit(X_tr,y_tr); joblib.dump(xm,"models/xgb_model.pkl"); metrics["XGBoost"]=ev("XGBoost",xm)

if HAS_TF:
    print("Training Deep Learning ANN…")
    dl = Sequential([Dense(256,"relu",input_shape=(X_tr.shape[1],)),BatchNormalization(),Dropout(.3),
                     Dense(128,"relu"),BatchNormalization(),Dropout(.3),
                     Dense(64,"relu"),Dropout(.2),Dense(32,"relu"),Dense(1,"sigmoid")])
    dl.compile(Adam(.001),"binary_crossentropy",["accuracy"])
    dl.fit(X_tr,y_tr,epochs=120,batch_size=64,validation_split=.1,
           callbacks=[EarlyStopping(patience=15,restore_best_weights=True)],verbose=0)
    dl.save("models/dl_model.h5")
    metrics["Deep Learning ANN"]=ev("Deep Learning ANN",None,dl.predict(X_te,verbose=0).flatten())

# ── Save meta ─────────────────────────────────────────────────────────────────
meta = {"metrics":metrics,"all_features":ALL_FEATURES,"required_features":REQUIRED_FEATURES,
        "optional_features":OPTIONAL_FEATURES,"categorical_cols":CATEGORICAL_COLS,
        "numeric_cols":NUMERIC_COLS,"target":TARGET}
joblib.dump(meta,"models/meta.pkl")
print(f"\n✅ Done! Models: {list(metrics.keys())}")
print("  Run: streamlit run app.py")
