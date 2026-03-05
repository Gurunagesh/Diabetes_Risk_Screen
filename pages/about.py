import streamlit as st


def show():
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>Smart Diabetes Detection using Integrated ML &amp; DL Algorithms</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview
    A complete end-to-end diabetes prediction system built on the
    **Diabetes Prediction in India Dataset** (Kaggle, ankushpanday1).
    The system handles **incomplete patient data** by auto-estimating missing
    optional fields using KNN Imputation, enabling predictions even when
    only 5 of 20 input features are provided.

    ---
    ## 📋 Dataset Features (20 total)

    | Category | Features |
    |----------|---------|
    | 🔴 Required | Age, Gender, BMI, Physical Activity, Family History of Diabetes |
    | 🩸 Clinical (Optional) | Glucose Level, HbA1c, BP Systolic, BP Diastolic, Cholesterol |
    | 🏃 Lifestyle (Optional) | Smoking, Alcohol, Diet Type, Stress Level, Sleep Duration |
    | 📍 Other (Optional) | Waist Circumference, Urban/Rural, Hypertension, Prev. Gestational Diabetes, Medication Use |

    ---
    ## 🧠 Models Used

    | Model | Type | Strength |
    |-------|------|----------|
    | Random Forest | ML — Ensemble | Robust, handles non-linearity |
    | XGBoost | ML — Gradient Boost | High accuracy, imbalance-aware |
    | SVM (RBF) | ML — Kernel | High-dimensional effectiveness |
    | Logistic Regression | ML — Linear | Interpretable baseline |
    | Gradient Boosting | ML — Ensemble | Strong sequential learner |
    | Deep Learning ANN | DL — Neural Net | Learns complex patterns |

    **Final result** = average ensemble probability across all models.

    ---
    ## 🔄 Missing Data Strategy

    | Field Type | Handling |
    |------------|----------|
    | Required (5) | App rejects submission if missing |
    | Optional (15) | **KNN Imputer (k=5)** fills gaps from similar patients |

    The KNN Imputer is fitted on training data only — zero data leakage.
    The UI shows exactly which fields were auto-estimated.

    ---
    ## ⚙️ Full Training Pipeline

    1. **Data generation / loading** from CSV
    2. **Label Encoding** of all categorical features
    3. **KNN Imputation** — fitted on full training set
    4. **StandardScaler** — zero mean, unit variance
    5. **SMOTE** — synthetic minority oversampling for class balance
    6. **Train 5 ML + 1 DL model** — saved as .pkl / .h5
    7. **Ensemble** — average probability → risk score

    ---
    ## 📁 Project Structure
    ```
    smart_diabetes_detection/
    ├── app.py                  ← Streamlit entry point
    ├── train_models.py         ← Train all models (run once)
    ├── generate_dataset.py     ← Synthetic data generator
    ├── requirements.txt
    ├── README.md
    ├── data/
    │   └── diabetes_india.csv  ← Place real Kaggle CSV here
    ├── pages/
    │   ├── home.py
    │   ├── predict.py          ← Core UI with required/optional form
    │   ├── performance.py
    │   ├── insights.py
    │   └── about.py
    ├── utils/
    │   └── predictor.py        ← Imputation + prediction engine
    └── models/                 ← Auto-created after training
        ├── knn_imputer.pkl
        ├── scaler.pkl
        ├── label_encoders.pkl
        ├── rf_model.pkl
        ├── xgb_model.pkl
        ├── svm_model.pkl
        ├── lr_model.pkl
        ├── gb_model.pkl
        ├── dl_model.h5
        └── meta.pkl
    ```

    ---
    ## ⚠️ Disclaimer
    > This system is for **educational and research purposes only**.
    > It does **not** provide medical diagnoses.
    > Always consult a qualified healthcare professional for clinical decisions.
    """)

    st.markdown(
        "<div style='text-align:center;color:#888;margin-top:2rem'>"
        "Built with ❤️ · Streamlit · scikit-learn · XGBoost · TensorFlow · Plotly"
        "</div>",
        unsafe_allow_html=True
    )
