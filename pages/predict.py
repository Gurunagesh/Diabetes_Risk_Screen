import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.predictor import (
    predict, REQUIRED_FEATURES, OPTIONAL_FEATURES,
    FEATURE_LABELS, NORMAL_RANGES, ALL_FEATURES
)


def show():
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Diabetes Risk Prediction</h1>
        <p>Provide required fields. Optional fields are <b>auto-estimated</b> via KNN if left blank.</p>
    </div>""", unsafe_allow_html=True)

    # Legend
    col_l, col_r = st.columns(2)
    col_l.markdown(
        '🔴 <span class="req-badge">REQUIRED</span> &nbsp; Must be provided for prediction',
        unsafe_allow_html=True
    )
    col_r.markdown(
        '🟢 <span class="opt-badge">OPTIONAL</span> &nbsp; Auto-estimated via KNN if skipped',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── REQUIRED FIELDS ───────────────────────────────────────────────────────
    st.markdown("### 🔴 Required Information")
    st.caption("These 5 fields are essential. All must be filled before prediction.")

    r1, r2, r3 = st.columns(3)
    with r1:
        age = st.number_input(
            "Age (years) 🔴", min_value=1, max_value=120, value=None,
            placeholder="e.g. 45",
            help="Patient's current age in years."
        )
        bmi = st.number_input(
            "BMI (kg/m²) 🔴", min_value=10.0, max_value=70.0, value=None,
            placeholder="e.g. 26.5",
            help=NORMAL_RANGES.get("BMI", "")
        )
    with r2:
        gender = st.selectbox(
            "Gender 🔴",
            options=["", "Male", "Female"],
            index=0,
            help="Patient's biological sex."
        )
        physical_activity = st.selectbox(
            "Physical Activity Level 🔴",
            options=["", "Sedentary", "Low", "Moderate", "High"],
            index=0,
            help="Sedentary: no exercise · Low: light 1-2x/week · Moderate: 3-4x/week · High: daily vigorous"
        )
    with r3:
        family_history = st.selectbox(
            "Family History of Diabetes 🔴",
            options=["", "Yes", "No"],
            index=0,
            help="Does any first-degree relative (parent/sibling) have diabetes?"
        )

    st.markdown("---")

    # ── OPTIONAL FIELDS ───────────────────────────────────────────────────────
    st.markdown("### 🟢 Optional Clinical & Lifestyle Details")
    st.caption(
        "Any fields left blank will be **intelligently estimated** using KNN Imputation "
        "(5 most similar patient profiles in training data). Providing values improves accuracy."
    )

    with st.expander("➕ Enter optional details for higher accuracy", expanded=False):

        st.markdown("#### 🩸 Clinical Measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            glucose = st.number_input(
                "Blood Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=None,
                placeholder="e.g. 105",
                help=NORMAL_RANGES.get("Glucose_Level", "")
            )
            bp_sys = st.number_input(
                "Systolic BP (mmHg)", min_value=0.0, max_value=250.0, value=None,
                placeholder="e.g. 120",
                help=NORMAL_RANGES.get("Blood_Pressure_Systolic", "")
            )
        with c2:
            hba1c = st.number_input(
                "HbA1c Level (%)", min_value=0.0, max_value=15.0, value=None,
                placeholder="e.g. 5.8",
                help=NORMAL_RANGES.get("HbA1c_Level", "")
            )
            bp_dia = st.number_input(
                "Diastolic BP (mmHg)", min_value=0.0, max_value=150.0, value=None,
                placeholder="e.g. 78",
                help=NORMAL_RANGES.get("Blood_Pressure_Diastolic", "")
            )
        with c3:
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)", min_value=0.0, max_value=500.0, value=None,
                placeholder="e.g. 185",
                help=NORMAL_RANGES.get("Cholesterol_Level", "")
            )
            waist = st.number_input(
                "Waist Circumference (cm)", min_value=0.0, max_value=200.0, value=None,
                placeholder="e.g. 88"
            )

        st.markdown("#### 🏃 Lifestyle & History")
        l1, l2, l3 = st.columns(3)
        with l1:
            smoking = st.selectbox(
                "Smoking Status", ["", "Never", "Former", "Current"]
            )
            alcohol = st.selectbox(
                "Alcohol Consumption", ["", "None", "Occasional", "Regular"]
            )
            diet_type = st.selectbox(
                "Diet Type", ["", "Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"]
            )
        with l2:
            stress = st.selectbox(
                "Stress Level", ["", "Low", "Moderate", "High"]
            )
            sleep = st.number_input(
                "Sleep Duration (hrs/day)", min_value=0.0, max_value=24.0, value=None,
                placeholder="e.g. 7",
                help=NORMAL_RANGES.get("Sleep_Duration", "")
            )
            urban_rural = st.selectbox(
                "Residence Type", ["", "Urban", "Rural"]
            )
        with l3:
            hypertension = st.selectbox(
                "Hypertension", ["", "Yes", "No"]
            )
            prev_gest = st.selectbox(
                "Prev. Gestational Diabetes",
                ["", "Yes", "No"],
                help="Applicable for female patients only."
            )
            medication = st.selectbox(
                "On Diabetes Medication", ["", "Yes", "No"]
            )

    # ── Helper: None if empty string ─────────────────────────────────────────
    def _val(v):
        return None if (v == "" or v is None) else v

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        # Validation
        errors = []
        if age is None:                    errors.append("Age")
        if not gender:                     errors.append("Gender")
        if bmi is None:                    errors.append("BMI")
        if not physical_activity:          errors.append("Physical Activity Level")
        if not family_history:             errors.append("Family History of Diabetes")

        if errors:
            st.error(f"⚠️ Please fill the required fields: **{', '.join(errors)}**")
            return

        user_inputs = {
            # Required
            "Age":                           age,
            "Gender":                        gender,
            "BMI":                           bmi,
            "Physical_Activity":             physical_activity,
            "Family_History_of_Diabetes":    family_history,
            # Optional
            "Glucose_Level":                 _val(glucose),
            "HbA1c_Level":                   _val(hba1c),
            "Blood_Pressure_Systolic":       _val(bp_sys),
            "Blood_Pressure_Diastolic":      _val(bp_dia),
            "Cholesterol_Level":             _val(cholesterol),
            "Smoking_Status":                _val(smoking),
            "Alcohol_Consumption":           _val(alcohol),
            "Diet_Type":                     _val(diet_type),
            "Stress_Level":                  _val(stress),
            "Sleep_Duration":                _val(sleep),
            "Waist_Circumference":           _val(waist),
            "Urban_Rural":                   _val(urban_rural),
            "Hypertension":                  _val(hypertension),
            "Previous_Gestational_Diabetes": _val(prev_gest),
            "Medication_Use":                _val(medication),
        }

        with st.spinner("🧠 Running ML & DL models…"):
            try:
                result = predict(user_inputs)
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                return
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return

        _display_result(result)


def _display_result(result):
    pred  = result["ensemble_prediction"]
    prob  = result["ensemble_probability"]
    level = result["risk_level"]

    # ── Main verdict ──────────────────────────────────────────────────────────
    if level == "High":
        st.markdown(f"""
        <div class="result-high">
            ⚠️ HIGH RISK — Diabetes Likely<br>
            <span style="font-size:1rem;font-weight:400">Ensemble probability: {prob}%</span>
        </div>""", unsafe_allow_html=True)
    elif level == "Moderate":
        st.markdown(f"""
        <div class="result-moderate">
            ⚡ MODERATE RISK — Pre-Diabetic Zone<br>
            <span style="font-size:1rem;font-weight:400">Ensemble probability: {prob}%</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            ✅ LOW RISK — No Diabetes Detected<br>
            <span style="font-size:1rem;font-weight:400">Ensemble probability: {prob}%</span>
        </div>""", unsafe_allow_html=True)

    st.caption("⚠️ Educational tool only — not a substitute for professional medical advice.")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge ─────────────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    gauge_color = "#e74c3c" if prob >= 65 else ("#e67e22" if prob >= 35 else "#27ae60")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={"reference": 50, "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#27ae60"}},
        title={"text": "Diabetes Risk Score (%)  |  Threshold: 50%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": gauge_color},
            "steps": [
                {"range": [0,  35], "color": "#d5f5e3"},
                {"range": [35, 65], "color": "#fdebd0"},
                {"range": [65, 100],"color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
        }
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-model breakdown ───────────────────────────────────────────────────
    st.markdown("### 📊 Individual Model Results")
    cols = st.columns(len(result["models"]))
    for i, (name, res) in enumerate(result["models"].items()):
        color = "#e74c3c" if res["prediction"] == 1 else "#27ae60"
        label = "DIABETIC" if res["prediction"] == 1 else "NON-DIABETIC"
        with cols[i]:
            st.markdown(f"""
            <div class="model-card">
                <b style="font-size:.85rem">{name}</b><br>
                <span style="color:{color};font-size:1rem;font-weight:700">{label}</span><br>
                <span style="font-size:.82rem;color:#555">Prob: {res['probability']}%</span>
            </div>""", unsafe_allow_html=True)

    # ── Probability bar chart ─────────────────────────────────────────────────
    import plotly.express as px
    import pandas as pd
    bar_df = pd.DataFrame([
        {"Model": k, "Probability (%)": v["probability"],
         "Outcome": "Diabetic" if v["prediction"] == 1 else "Non-Diabetic"}
        for k, v in result["models"].items()
    ])
    fig2 = px.bar(bar_df, x="Model", y="Probability (%)", color="Outcome",
                  color_discrete_map={"Diabetic": "#e74c3c", "Non-Diabetic": "#27ae60"},
                  title="Prediction Probability per Model")
    fig2.add_hline(y=50, line_dash="dash", line_color="black",
                   annotation_text="Decision threshold (50%)")
    fig2.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Imputed fields notice ─────────────────────────────────────────────────
    if result["imputed_fields"]:
        from utils.predictor import FEATURE_LABELS
        rows_html = "".join(
            f"&nbsp;&nbsp;• <b>{FEATURE_LABELS.get(k, k)}</b>: {v}<br>"
            for k, v in result["imputed_fields"].items()
        )
        st.markdown(f"""
        <div class="imputed-box">
            <b>🔄 Auto-Estimated Fields (KNN Imputation, k=5)</b><br><br>
            The following optional fields were not provided and were automatically
            estimated from the 5 most similar patients in the training data:<br><br>
            {rows_html}
            <br><i>💡 Entering these values manually will improve prediction accuracy.</i>
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ All fields were provided — no imputation needed. Maximum accuracy achieved.")
