import streamlit as st
import pandas as pd


def show():
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Smart Diabetes Detection System</h1>
        <p>India Diabetes Dataset · Integrated ML &amp; DL Algorithms · Smart Imputation</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "🧠", "6 Models",       "ML + DL Ensemble"),
        (c2, "🔄", "KNN (k=5)",      "Smart Imputation"),
        (c3, "📊", "20 Features",    "Clinical + Lifestyle"),
        (c4, "🎯", "5 Required",     "15 Optional"),
    ]:
        col.markdown(f"""
        <div class="section-card" style="text-align:center; border-left:4px solid #1a5276;">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-size:1.4rem;font-weight:700;color:#1a5276">{val}</div>
            <div style="font-size:0.85rem;color:#555">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 🎯 Required vs Optional Fields")
        data = {
            "Feature": [
                "Age", "Gender", "BMI",
                "Physical Activity", "Family History of Diabetes",
                "Blood Glucose Level", "HbA1c Level",
                "Blood Pressure (Systolic)", "Blood Pressure (Diastolic)",
                "Cholesterol Level", "Smoking Status",
                "Alcohol Consumption", "Diet Type", "Stress Level",
                "Sleep Duration", "Waist Circumference",
                "Urban / Rural", "Hypertension",
                "Prev. Gestational Diabetes", "Medication Use",
            ],
            "Status": (
                ["🔴 Required"] * 5 +
                ["🟢 Optional"] * 15
            ),
            "If Missing": (
                ["❌ Must provide"] * 5 +
                ["✅ Auto-estimated"] * 15
            ),
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     height=560)

    with col_right:
        st.markdown("### 🚀 How to Use")
        st.info("""
        1. Go to **🔬 Predict** in the sidebar
        2. Fill in the **5 required fields**:
           - Age, Gender, BMI
           - Physical Activity, Family History
        3. Optionally add clinical details for more accuracy
        4. Click **Predict** — get results from 6 models + ensemble risk score
        """)

        st.markdown("### 🔁 Prediction Pipeline")
        st.markdown("""
        ```
        User Input
             │
             ├─ Required missing? ──► ❌ Error
             │
             └─ Optional missing? ──► ✅ KNN fill
                      │
                   Encode  →  Scale
                      │
             RF / XGB / SVM / LR / GB / DL
                      │
              Ensemble average → Risk Score
        ```
        """)

        st.markdown("### ⚠️ Disclaimer")
        st.warning(
            "This tool is for **educational purposes only**. "
            "It does not constitute medical advice. "
            "Always consult a qualified healthcare professional."
        )
