import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os


def _load_data():
    """Load real or synthetic dataset for visualization."""
    csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "diabetes_india.csv"
    )
    if os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        # Mini synthetic fallback
        np.random.seed(7)
        n = 2000
        age = np.random.randint(18, 80, n).astype(float)
        bmi = np.round(np.random.normal(25.5, 5.5, n), 1).clip(13, 55)
        glucose = np.round(np.random.normal(110, 32, n), 1).clip(60, 300)
        hba1c   = np.round(np.random.normal(5.6, 1.1, n), 1).clip(3.5, 12)
        pa      = np.random.choice(["Sedentary","Low","Moderate","High"], n)
        fh      = np.random.choice(["Yes","No"], n, p=[0.35, 0.65])
        gender  = np.random.choice(["Male","Female"], n)
        risk = ((glucose>140)*2 + (hba1c>6.5)*2 + (bmi>30) +
                (age>45) + (fh=="Yes").astype(int))
        outcome = (risk >= 4).astype(int)
        df = pd.DataFrame({
            "Age": age, "BMI": bmi, "Glucose_Level": glucose,
            "HbA1c_Level": hba1c, "Physical_Activity": pa,
            "Family_History_of_Diabetes": fh, "Gender": gender,
            "Diabetic": outcome,
        })
    df["Diagnosis"] = df["Diabetic"].map({0: "Non-Diabetic", 1: "Diabetic"})
    return df


COLORS = {"Non-Diabetic": "#27ae60", "Diabetic": "#e74c3c"}


def show():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Data Insights</h1>
        <p>Visual exploration of diabetes risk factors — India Dataset</p>
    </div>""", unsafe_allow_html=True)

    df = _load_data()

    # ── Top stats ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records",   f"{len(df):,}")
    s2.metric("Diabetic",        f"{df['Diabetic'].sum():,}")
    s3.metric("Non-Diabetic",    f"{(1-df['Diabetic']).sum():,}")
    s4.metric("Prevalence",      f"{df['Diabetic'].mean()*100:.1f}%")

    st.markdown("---")

    # ── Row 1: Glucose + HbA1c ────────────────────────────────────────────────
    if "Glucose_Level" in df.columns and "HbA1c_Level" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Glucose_Level", color="Diagnosis",
                               barmode="overlay", opacity=0.75,
                               title="Blood Glucose Distribution",
                               color_discrete_map=COLORS,
                               labels={"Glucose_Level": "Glucose (mg/dL)"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="HbA1c_Level", color="Diagnosis",
                               barmode="overlay", opacity=0.75,
                               title="HbA1c Distribution",
                               color_discrete_map=COLORS,
                               labels={"HbA1c_Level": "HbA1c (%)"})
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: BMI + Age ──────────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="Diagnosis", y="BMI", color="Diagnosis",
                     title="BMI Distribution by Diagnosis",
                     color_discrete_map=COLORS)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(df, x="Age", color="Diagnosis",
                           barmode="overlay", opacity=0.75,
                           title="Age Distribution by Diagnosis",
                           color_discrete_map=COLORS,
                           labels={"Age": "Age (years)"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Physical Activity + Family History ─────────────────────────────
    if "Physical_Activity" in df.columns:
        c5, c6 = st.columns(2)
        with c5:
            pa_counts = (
                df.groupby(["Physical_Activity", "Diagnosis"])
                .size().reset_index(name="Count")
            )
            fig = px.bar(pa_counts, x="Physical_Activity", y="Count",
                         color="Diagnosis", barmode="group",
                         title="Physical Activity vs Diabetes",
                         color_discrete_map=COLORS,
                         category_orders={"Physical_Activity":
                             ["Sedentary", "Low", "Moderate", "High"]})
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            if "Family_History_of_Diabetes" in df.columns:
                fh_counts = (
                    df.groupby(["Family_History_of_Diabetes", "Diagnosis"])
                    .size().reset_index(name="Count")
                )
                fig = px.bar(fh_counts, x="Family_History_of_Diabetes", y="Count",
                             color="Diagnosis", barmode="group",
                             title="Family History vs Diabetes",
                             color_discrete_map=COLORS)
                st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Gender pie + Scatter ───────────────────────────────────────────
    c7, c8 = st.columns(2)
    with c7:
        vc = df["Diagnosis"].value_counts().reset_index()
        vc.columns = ["Diagnosis", "Count"]
        fig = px.pie(vc, names="Diagnosis", values="Count",
                     color="Diagnosis", color_discrete_map=COLORS,
                     title="Dataset Class Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c8:
        if "Glucose_Level" in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), x="BMI", y="Glucose_Level",
                             color="Diagnosis", opacity=0.6,
                             title="BMI vs Glucose (sampled)",
                             color_discrete_map=COLORS,
                             trendline="ols",
                             labels={"Glucose_Level": "Glucose (mg/dL)"})
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("### 🔥 Feature Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        corr = num_df.corr().round(2)
        fig_h = px.imshow(corr, text_auto=True, aspect="auto",
                          color_continuous_scale="RdBu_r",
                          title="Pearson Correlation — Numeric Features")
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Gender breakdown ──────────────────────────────────────────────────────
    if "Gender" in df.columns:
        st.markdown("### 👥 Gender-wise Diabetes Prevalence")
        g_df = (df.groupby(["Gender","Diagnosis"])
                  .size().reset_index(name="Count"))
        fig_g = px.bar(g_df, x="Gender", y="Count", color="Diagnosis",
                       barmode="group", color_discrete_map=COLORS,
                       title="Diabetes by Gender")
        st.plotly_chart(fig_g, use_container_width=True)
