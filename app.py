import streamlit as st

st.set_page_config(
    page_title="Smart Diabetes Detection — India",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B4F72 60%, #0D1B2A 100%);
    padding: 2rem 2.5rem; border-radius: 16px;
    margin-bottom: 1.8rem; color: white; text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.8; margin-top: 0.4rem; }

.req-badge  { background:#e74c3c; color:#fff; padding:2px 9px;
              border-radius:12px; font-size:0.72rem; font-weight:600; }
.opt-badge  { background:#27ae60; color:#fff; padding:2px 9px;
              border-radius:12px; font-size:0.72rem; font-weight:600; }

.result-high {
    background: linear-gradient(135deg,#c0392b,#e74c3c);
    color:#fff; padding:1.8rem; border-radius:14px;
    text-align:center; font-size:1.6rem; font-weight:700;
    box-shadow:0 6px 20px rgba(231,76,60,.4);
}
.result-moderate {
    background: linear-gradient(135deg,#d35400,#e67e22);
    color:#fff; padding:1.8rem; border-radius:14px;
    text-align:center; font-size:1.6rem; font-weight:700;
    box-shadow:0 6px 20px rgba(230,126,34,.4);
}
.result-low {
    background: linear-gradient(135deg,#1e8449,#27ae60);
    color:#fff; padding:1.8rem; border-radius:14px;
    text-align:center; font-size:1.6rem; font-weight:700;
    box-shadow:0 6px 20px rgba(39,174,96,.4);
}
.model-card {
    background:#f8fbff; border-radius:10px; padding:1rem;
    text-align:center; border:1.5px solid #d6eaf8;
    box-shadow:0 2px 8px rgba(0,0,0,.06);
}
.imputed-box {
    background:#fffbeb; border:1.5px solid #f0c040;
    border-radius:10px; padding:1rem; margin-top:1rem;
}
.section-card {
    background:#fff; border-radius:12px; padding:1.4rem;
    box-shadow:0 3px 12px rgba(0,0,0,.08);
    border-left:4px solid #2980b9; margin-bottom:1rem;
}
.stButton > button {
    background:linear-gradient(135deg,#1a5276,#2980b9);
    color:white; border:none; border-radius:30px;
    padding:.65rem 2rem; font-size:1.1rem;
    font-weight:600; width:100%; letter-spacing:.5px;
    transition: transform .2s, box-shadow .2s;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 6px 18px rgba(41,128,185,.45);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/stethoscope.png", width=60
)
st.sidebar.markdown("## 🩺 Smart Diabetes\nDetection System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔬 Predict",
    "📊 Model Performance",
    "📈 Data Insights",
    "ℹ️ About",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: Diabetes Prediction in India\n"
    "Source: Kaggle — ankushpanday1"
)

# ── Route pages ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages.home import show
elif page == "🔬 Predict":
    from pages.predict import show
elif page == "📊 Model Performance":
    from pages.performance import show
elif page == "📈 Data Insights":
    from pages.insights import show
else:
    from pages.about import show

show()
