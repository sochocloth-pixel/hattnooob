"""
CropGuard – AI-Based Crop Health Prediction System
Public Web Version (Streamlit) – Modern Figma-style UI

Libraries: Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit
CBSE Class 12 level – Python only
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CropGuard | Crop Health Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- FIGMA / CANVA STYLE CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #F4F7F2 0%, #E8F0E6 40%, #F7F5F0 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------- Main container spacing ---------- */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ---------- Hero Header ---------- */
.hero {
    background: linear-gradient(135deg, #0D3B2E 0%, #1B5E3B 45%, #2E7D4F 100%);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 12px 40px rgba(13, 59, 46, 0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    opacity: 0.88;
    margin: 0.45rem 0 0 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 0.28rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.3px;
}

/* ---------- Cards ---------- */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.03);
    border: 1px solid rgba(0,0,0,0.04);
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6B7C6E;
    margin: 0 0 0.9rem 0;
}

/* ---------- Result states ---------- */
.result-box {
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin: 0.5rem 0 1.2rem 0;
}
.result-healthy {
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    border: 1px solid #A5D6A7;
}
.result-moderate {
    background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
    border: 1px solid #FFE082;
}
.result-poor {
    background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
    border: 1px solid #EF9A9A;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    opacity: 0.7;
    margin: 0;
}
.result-status {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.25rem 0 0.5rem 0;
    letter-spacing: -0.4px;
}
.result-desc {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    line-height: 1.45;
}

/* ---------- Recommendation ---------- */
.rec-box {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border-left: 4px solid #2E7D4F;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 1.2rem;
}
.rec-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #2E7D4F;
    margin: 0 0 0.3rem 0;
}
.rec-text {
    font-size: 0.92rem;
    color: #2C2C2C;
    margin: 0;
    line-height: 1.45;
}

/* ---------- Input summary chips ---------- */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 1rem 0;
}
.chip {
    background: #F0F4F0;
    border-radius: 50px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1B5E3B;
    border: 1px solid #D4E0D4;
}

/* ---------- Sidebar polish ---------- */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8EDE8;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}
.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0D3B2E;
    letter-spacing: -0.4px;
    margin-bottom: 0.15rem;
}
.sidebar-tag {
    font-size: 0.78rem;
    color: #6B7C6E;
    margin-bottom: 1.4rem;
}

/* Metric cards in sidebar */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.stat-item {
    background: #F4F7F2;
    border-radius: 10px;
    padding: 0.7rem 0.4rem;
    text-align: center;
    border: 1px solid #E0E8E0;
}
.stat-value {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0D3B2E;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #6B7C6E;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.15rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #1B5E3B 0%, #2E7D4F 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 16px rgba(27, 94, 59, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(27, 94, 59, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ---------- Number inputs ---------- */
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #D4E0D4 !important;
    padding: 0.55rem 0.8rem !important;
    font-weight: 500 !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #2E7D4F !important;
    box-shadow: 0 0 0 3px rgba(46, 125, 79, 0.15) !important;
}

/* ---------- Section headers ---------- */
.section-h {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0D3B2E;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.3px;
}

/* ---------- How it works ---------- */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
}
.step-card {
    background: #F8FAF7;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #E0E8E0;
    text-align: center;
}
.step-num {
    width: 28px;
    height: 28px;
    background: #1B5E3B;
    color: white;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.step-text {
    font-size: 0.78rem;
    color: #3D4F40;
    line-height: 1.35;
    margin: 0;
}

/* ---------- Footer ---------- */
.footer-bar {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
    color: #8A9A8C;
    font-size: 0.8rem;
    font-weight: 500;
}
.footer-bar span {
    color: #1B5E3B;
    font-weight: 700;
}

/* ---------- Placeholder state ---------- */
.placeholder {
    background: #FFFFFF;
    border: 2px dashed #D4E0D4;
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: #8A9A8C;
}
.placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD & TRAIN MODEL (cached) --------------------
@st.cache_resource
def load_and_train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "crop_health_data.csv")

    if not os.path.exists(csv_path):
        st.error("Dataset file 'crop_health_data.csv' not found.")
        st.stop()

    df = pd.read_csv(csv_path)
    df = df.dropna()

    expected = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    if list(df.columns) != expected:
        if len(df.columns) == 4:
            df.columns = expected
        else:
            st.error("CSV must have columns: Humidity, Temperature, Rainfall, Crop_Health")
            st.stop()

    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, df


model, accuracy, df = load_and_train_model()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌱 CropGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">AI Crop Health Prediction</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🌤 Environmental Inputs")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=55.0,
        value=28.0, step=0.5,
        help="Typical range for most crops: 15–40 °C"
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0,
        value=65.0, step=1.0,
        help="Relative humidity (0–100%)"
    )
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=400.0,
        value=120.0, step=5.0,
        help="Rainfall amount in millimetres"
    )

    st.markdown("")
    predict_btn = st.button("🌿  Predict Crop Health", use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📊 Model Info")

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Records</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{accuracy*100:.0f}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Decision Tree • max_depth = 6")


# -------------------- HERO HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">🌱 CropGuard</div>
    <div class="hero-sub">Predict crop health from temperature, humidity & rainfall using machine learning</div>
    <div class="hero-badge">● Model Ready &nbsp;·&nbsp; Decision Tree Classifier</div>
</div>
""", unsafe_allow_html=True)


# -------------------- MAIN LAYOUT --------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<p class="section-h">Crop Health Status</p>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame(
            [[humidity, temperature, rainfall]],
            columns=["Humidity", "Temperature", "Rainfall"]
        )
        prediction = model.predict(input_df)[0]

        if prediction == "Healthy":
            status_color = "#1B5E3B"
            box_class = "result-healthy"
            emoji = "🟢"
            desc = "The current environmental conditions are favorable for crop growth."
            rec = "Conditions appear favourable. Continue regular irrigation and crop monitoring."
        elif prediction == "Moderately Healthy":
            status_color = "#E65100"
            box_class = "result-moderate"
            emoji = "🟡"
            desc = "Environmental conditions are somewhat suitable for the crop."
            rec = "Monitor soil moisture and temperature regularly. Adjust irrigation if needed."
        else:
            status_color = "#C62828"
            box_class = "result-poor"
            emoji = "🔴"
            desc = "Conditions may be unfavourable for healthy crop growth."
            rec = "Check irrigation, temperature stress and rainfall conditions carefully."

        st.markdown(f"""
        <div class="result-box {box_class}">
            <p class="result-label">Prediction Result</p>
            <p class="result-status" style="color:{status_color};">{emoji} {prediction.upper()}</p>
            <p class="result-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="chip-row">
            <span class="chip">🌡 {temperature} °C</span>
            <span class="chip">💧 {humidity} %</span>
            <span class="chip">🌧 {rainfall} mm</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rec-box">
            <p class="rec-title">💡 Recommendation</p>
            <p class="rec-text">{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        # ---- Bar chart ----
        st.markdown('<p class="section-h">Current Environmental Conditions</p>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(5.8, 3.0), facecolor="white")
        labels = ["Temperature\n(°C)", "Humidity\n(%)", "Rainfall\n(mm)"]
        values = [temperature, humidity, rainfall]
        colors = ["#1B5E3B", "#43A047", "#F9A825"]
        bars = ax1.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
        ax1.set_ylabel("Value", fontsize=9, color="#555")
        ax1.set_ylim(0, max(values) * 1.3 + 5)
        ax1.tick_params(colors="#555", labelsize=9)
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                f"{val}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1B5E3B"
            )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_color("#DDD")
        ax1.spines["bottom"].set_color("#DDD")
        ax1.set_facecolor("#FAFCFA")
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    else:
        st.markdown("""
        <div class="placeholder">
            <div class="placeholder-icon">🌿</div>
            <p style="font-weight:600; margin:0 0 0.3rem 0; color:#3D4F40;">Ready to predict</p>
            <p style="margin:0; font-size:0.88rem;">Enter temperature, humidity & rainfall<br>in the sidebar, then click <b>Predict</b></p>
        </div>
        """, unsafe_allow_html=True)


with right:
    st.markdown('<p class="section-h">Dataset Overview</p>', unsafe_allow_html=True)

    # Scatter plot
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    colour_map = {
        "Healthy": "#2E7D4F",
        "Moderately Healthy": "#F9A825",
        "Poor Health": "#E53935"
    }
    for label, colour in colour_map.items():
        subset = df[df["Crop_Health"] == label]
        ax2.scatter(
            subset["Temperature"], subset["Humidity"],
            c=colour, label=label, alpha=0.75, s=36,
            edgecolors="white", linewidths=0.5
        )
    ax2.set_xlabel("Temperature (°C)", fontsize=9, color="#555")
    ax2.set_ylabel("Humidity (%)", fontsize=9, color="#555")
    ax2.tick_params(colors="#555", labelsize=8)
    ax2.legend(fontsize=7.5, loc="best", framealpha=0.95, edgecolor="#EEE")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#DDD")
    ax2.spines["bottom"].set_color("#DDD")
    ax2.set_facecolor("#FAFCFA")
    ax2.set_title("Temperature vs Humidity", fontsize=11, fontweight="bold",
                  color="#0D3B2E", pad=8)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Class distribution
    st.markdown('<p class="section-h" style="margin-top:0.8rem;">Class Distribution</p>', unsafe_allow_html=True)
    class_counts = df["Crop_Health"].value_counts()
    st.bar_chart(class_counts, color="#2E7D4F", height=180)


# -------------------- HOW IT WORKS --------------------
st.markdown("---")
st.markdown('<p class="section-h">How It Works</p>', unsafe_allow_html=True)

st.markdown("""
<div class="steps-grid">
    <div class="step-card">
        <div class="step-num">1</div>
        <p class="step-text">Load dataset with <b>Pandas</b></p>
    </div>
    <div class="step-card">
        <div class="step-num">2</div>
        <p class="step-text">Select features: Temp, Humidity, Rainfall</p>
    </div>
    <div class="step-card">
        <div class="step-num">3</div>
        <p class="step-text">Split into train & test sets</p>
    </div>
    <div class="step-card">
        <div class="step-num">4</div>
        <p class="step-text">Train <b>Decision Tree</b> model</p>
    </div>
    <div class="step-card">
        <div class="step-num">5</div>
        <p class="step-text">User enters values → Predict</p>
    </div>
    <div class="step-card">
        <div class="step-num">6</div>
        <p class="step-text">Show result + <b>Matplotlib</b> charts</p>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer-bar">
    <span>CropGuard</span> · CBSE Class 12 Computer Science Project<br>
    Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
