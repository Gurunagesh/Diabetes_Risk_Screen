import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib, os


def show():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance Dashboard</h1>
        <p>Accuracy, F1, AUC and confusion matrices for all 6 algorithms</p>
    </div>""", unsafe_allow_html=True)

    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "meta.pkl"
    )
    if not os.path.exists(meta_path):
        st.warning("⚠️ Models not trained yet. Run `python train_models.py` first.")
        st.code("python train_models.py", language="bash")
        return

    meta    = joblib.load(meta_path)
    metrics = meta["metrics"]

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Performance Summary")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{m['accuracy']}%",
            "Precision": f"{m['precision']}%",
            "Recall":    f"{m['recall']}%",
            "F1 Score":  f"{m['f1']}%",
            "ROC-AUC":   f"{m['roc_auc']}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    best = max(metrics.items(), key=lambda x: x[1]["roc_auc"])
    st.success(f"🏆 Best model (ROC-AUC): **{best[0]}** — {best[1]['roc_auc']}%")

    st.markdown("---")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("### 📊 Metric Comparison")
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = {"accuracy": "Accuracy", "precision": "Precision",
                     "recall": "Recall", "f1": "F1 Score", "roc_auc": "ROC-AUC"}
    plot_rows = []
    for name, m in metrics.items():
        for k in metric_keys:
            plot_rows.append({"Model": name, "Metric": metric_labels[k], "Score (%)": m[k]})
    df_plot = pd.DataFrame(plot_rows)

    fig = px.bar(df_plot, x="Metric", y="Score (%)", color="Model",
                 barmode="group", title="All Models — All Metrics",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("### 🕸️ Radar Chart")
    fig2 = go.Figure()
    for name, m in metrics.items():
        vals = [m[k] for k in metric_keys] + [m[metric_keys[0]]]
        fig2.add_trace(go.Scatterpolar(
            r=vals,
            theta=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Accuracy"],
            fill="toself", name=name
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[70, 100])),
        height=440, title="Model Performance Radar"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### 🔲 Confusion Matrices")
    n = len(metrics)
    cols_per_row = 3
    items = list(metrics.items())
    for row_start in range(0, n, cols_per_row):
        cols = st.columns(min(cols_per_row, n - row_start))
        for ci, (name, m) in enumerate(items[row_start:row_start+cols_per_row]):
            with cols[ci]:
                fig3 = px.imshow(
                    m["cm"], text_auto=True,
                    x=["Non-Diabetic", "Diabetic"],
                    y=["Non-Diabetic", "Diabetic"],
                    labels=dict(x="Predicted", y="Actual"),
                    title=name, color_continuous_scale="Blues"
                )
                fig3.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
                st.plotly_chart(fig3, use_container_width=True)
