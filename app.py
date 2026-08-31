"""
CropGuard – Crop Health Prediction System
Clean, readable UI with Dark/Light mode
Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
CBSE Class 12
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

# -------------------- SESSION STATE --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "pred_inputs" not in st.session_state:
    st.session_state.pred_inputs = None

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_and_train_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    if not os.path.exists(path):
        st.error("crop_health_data.csv not found")
        st.stop()
    df = pd.read_csv(path).dropna()
    if list(df.columns) != ["Humidity", "Temperature", "Rainfall", "Crop_Health"]:
        if len(df.columns) == 4:
            df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, df

model, accuracy, df = load_and_train_model()
is_dark = st.session_state.theme == "dark"

# -------------------- CLEAN CSS (high contrast) --------------------
if is_dark:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #111814 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; max-width: 1100px; }

    section[data-testid="stSidebar"] {
        background: #1A221C !important;
        border-right: 1px solid #2A352C !important;
    }
    section[data-testid="stSidebar"] * { color: #E8EDE9 !important; }
    section[data-testid="stSidebar"] label { color: #A8B8AC !important; font-weight: 600 !important; }

    .hero {
        background: linear-gradient(135deg, #0D3B2E, #1B5E3B);
        border-radius: 16px; padding: 1.6rem 2rem; color: #FFFFFF;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; color: #FFFFFF; }
    .hero p { margin: 0.35rem 0 0; opacity: 0.9; color: #D0E8D8; font-size: 0.95rem; }

    .result-card {
        border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .result-healthy { background: #1B3D2F; border: 2px solid #3DDC84; }
    .result-moderate { background: #3D3010; border: 2px solid #F9A825; }
    .result-poor { background: #3D1A1A; border: 2px solid #EF5350; }
    .result-card h2 { margin: 0.2rem 0 0.4rem; font-size: 1.6rem; font-weight: 800; }
    .result-card p { margin: 0; color: #C8D4CC; font-size: 0.92rem; }

    .info-box {
        background: #1A221C; border: 1px solid #2A352C; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 1rem; color: #E8EDE9;
    }
    .info-box strong { color: #3DDC84; }

    .chip {
        display: inline-block; background: #243028; border: 1px solid #3A453C;
        border-radius: 20px; padding: 0.3rem 0.85rem; margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.85rem; font-weight: 600; color: #3DDC84;
    }

    .section-title {
        color: #E8EDE9 !important; font-size: 1.05rem; font-weight: 700;
        margin: 0 0 0.8rem 0;
    }

    .stButton > button {
        background: #2E7D4F !important; color: #FFFFFF !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; padding: 0.6rem 1.2rem !important;
    }
    .stButton > button:hover { background: #3DDC84 !important; color: #0D3B2E !important; }

    .stNumberInput input {
        background: #121812 !important; color: #E8EDE9 !important;
        border: 1.5px solid #2A352C !important; border-radius: 8px !important;
    }

    div[data-testid="stMetricValue"] { color: #3DDC84 !important; }
    div[data-testid="stMetricLabel"] { color: #A8B8AC !important; }

    hr { border-color: #2A352C !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: #F5F7F4 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem !important; max-width: 1100px; }

    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #DCE5DC !important;
    }
    section[data-testid="stSidebar"] * { color: #1A2E22 !important; }
    section[data-testid="stSidebar"] label { color: #4A5C50 !important; font-weight: 600 !important; }

    .hero {
        background: linear-gradient(135deg, #0D3B2E, #1B5E3B);
        border-radius: 16px; padding: 1.6rem 2rem; color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 24px rgba(13,59,46,0.18);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; color: #FFFFFF; }
    .hero p { margin: 0.35rem 0 0; opacity: 0.9; color: #D0E8D8; font-size: 0.95rem; }

    .result-card {
        border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }
    .result-healthy { background: #E8F5E9; border: 2px solid #2E7D4F; }
    .result-moderate { background: #FFF8E1; border: 2px solid #F9A825; }
    .result-poor { background: #FFEBEE; border: 2px solid #C62828; }
    .result-card h2 { margin: 0.2rem 0 0.4rem; font-size: 1.6rem; font-weight: 800; }
    .result-card p { margin: 0; color: #333; font-size: 0.92rem; }

    .info-box {
        background: #FFFFFF; border: 1px solid #DCE5DC; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 1rem; color: #1A2E22;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .info-box strong { color: #1B5E3B; }

    .chip {
        display: inline-block; background: #E8F0E8; border: 1px solid #C5D5C5;
        border-radius: 20px; padding: 0.3rem 0.85rem; margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.85rem; font-weight: 600; color: #1B5E3B;
    }

    .section-title {
        color: #0D3B2E !important; font-size: 1.05rem; font-weight: 700;
        margin: 0 0 0.8rem 0;
    }

    .stButton > button {
        background: #1B5E3B !important; color: #FFFFFF !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; padding: 0.6rem 1.2rem !important;
    }
    .stButton > button:hover { background: #2E7D4F !important; }

    .stNumberInput input {
        background: #FFFFFF !important; color: #1A2E22 !important;
        border: 1.5px solid #C5D5C5 !important; border-radius: 8px !important;
    }

    div[data-testid="stMetricValue"] { color: #1B5E3B !important; }
    div[data-testid="stMetricLabel"] { color: #4A5C50 !important; }

    hr { border-color: #DCE5DC !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("### 🌱 CropGuard")
    with c2:
        if st.button("🌙" if not is_dark else "☀️", help="Toggle theme"):
            st.session_state.theme = "dark" if not is_dark else "light"
            st.rerun()

    st.markdown("---")
    st.markdown("**Environmental Conditions**")

    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.0, step=5.0)

    st.markdown("")
    if st.button("🌿 Predict Crop Health", use_container_width=True):
        # Validation
        if humidity < 0 or humidity > 100:
            st.error("Humidity must be 0–100")
        elif rainfall < 0:
            st.error("Rainfall cannot be negative")
        else:
            inp = pd.DataFrame([[humidity, temperature, rainfall]],
                               columns=["Humidity", "Temperature", "Rainfall"])
            pred = model.predict(inp)[0]
            st.session_state.prediction = pred
            st.session_state.pred_inputs = (temperature, humidity, rainfall)

    st.markdown("---")
    st.markdown("**Dataset & Model**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Records", len(df))
    m2.metric("Features", 3)
    m3.metric("Accuracy", f"{accuracy*100:.0f}%")
    st.caption("DecisionTreeClassifier · max_depth=6")

# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
    <h1>🌱 CropGuard</h1>
    <p>AI-based crop health prediction using temperature, humidity and rainfall</p>
</div>
""", unsafe_allow_html=True)

# -------------------- MAIN CONTENT --------------------
left, right = st.columns([1.2, 1], gap="medium")

# Chart colours based on theme
if is_dark:
    chart_bg = "#1A221C"
    chart_text = "#C8D4CC"
    chart_spine = "#3A453C"
    title_color = "#3DDC84"
    bar_colors = ["#2E7D4F", "#43A047", "#F9A825"]
    scatter_colors = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    dist_color = "#3DDC84"
else:
    chart_bg = "#FFFFFF"
    chart_text = "#333333"
    chart_spine = "#CCCCCC"
    title_color = "#0D3B2E"
    bar_colors = ["#1B5E3B", "#43A047", "#F9A825"]
    scatter_colors = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    dist_color = "#2E7D4F"

with left:
    st.markdown('<p class="section-title">Crop Health Status</p>', unsafe_allow_html=True)

    if st.session_state.prediction:
        pred = st.session_state.prediction
        t, h, r = st.session_state.pred_inputs

        if pred == "Healthy":
            cls = "result-healthy"
            color = "#2E7D4F" if not is_dark else "#3DDC84"
            desc = "Environmental conditions are favorable for crop growth."
            rec = "Continue regular irrigation and crop monitoring."
        elif pred == "Moderately Healthy":
            cls = "result-moderate"
            color = "#E65100" if not is_dark else "#F9A825"
            desc = "Conditions are somewhat suitable. Monitor closely."
            rec = "Check soil moisture and temperature regularly."
        else:
            cls = "result-poor"
            color = "#C62828" if not is_dark else "#EF5350"
            desc = "Conditions may be unfavourable for healthy growth."
            rec = "Review irrigation, temperature stress and rainfall."

        st.markdown(f"""
        <div class="result-card {cls}">
            <p style="margin:0;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:0.7;">Prediction Result</p>
            <h2 style="color:{color};">● {pred.upper()}</h2>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <span class="chip">🌡 {t} °C</span>
        <span class="chip">💧 {h} %</span>
        <span class="chip">🌧 {r} mm</span>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="margin-top:0.8rem;">
            <strong>💡 Recommendation</strong><br>{rec}
        </div>
        """, unsafe_allow_html=True)

        # Bar chart – current inputs
        st.markdown('<p class="section-title">Current Conditions</p>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))
        fig1.patch.set_facecolor(chart_bg)
        ax1.set_facecolor(chart_bg)
        labels = ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]
        values = [t, h, r]
        bars = ax1.bar(labels, values, color=bar_colors, width=0.5)
        ax1.set_ylabel("Value", color=chart_text, fontsize=9)
        ax1.tick_params(colors=chart_text, labelsize=8)
        ax1.set_ylim(0, max(values) * 1.35 + 5)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                     str(val), ha="center", fontsize=9, fontweight="bold", color=title_color)
        for spine in ax1.spines.values():
            spine.set_color(chart_spine)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    else:
        st.info("Enter Temperature, Humidity and Rainfall in the sidebar, then click **Predict Crop Health**.")

with right:
    st.markdown('<p class="section-title">Dataset: Temperature vs Humidity</p>', unsafe_allow_html=True)

    # Scatter plot – always shows (uses full dataset)
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.5))
    fig2.patch.set_facecolor(chart_bg)
    ax2.set_facecolor(chart_bg)
    for label, colour in scatter_colors.items():
        subset = df[df["Crop_Health"] == label]
        ax2.scatter(subset["Temperature"], subset["Humidity"],
                    c=colour, label=label, alpha=0.8, s=30, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Temperature (°C)", color=chart_text, fontsize=9)
    ax2.set_ylabel("Humidity (%)", color=chart_text, fontsize=9)
    ax2.tick_params(colors=chart_text, labelsize=8)
    leg = ax2.legend(fontsize=7.5, loc="best", framealpha=0.95)
    leg.get_frame().set_facecolor(chart_bg)
    leg.get_frame().set_edgecolor(chart_spine)
    for text in leg.get_texts():
        text.set_color(chart_text)
    for spine in ax2.spines.values():
        spine.set_color(chart_spine)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Training Data Distribution", color=title_color, fontsize=11, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Class distribution – matplotlib bar (not st.bar_chart) so theme works
    st.markdown('<p class="section-title">Class Distribution</p>', unsafe_allow_html=True)
    counts = df["Crop_Health"].value_counts()
    fig3, ax3 = plt.subplots(figsize=(5.2, 2.4))
    fig3.patch.set_facecolor(chart_bg)
    ax3.set_facecolor(chart_bg)
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [counts.get(c, 0) for c in order]
    cols = [scatter_colors[c] for c in order]
    ax3.barh(order, vals, color=cols, height=0.55)
    ax3.tick_params(colors=chart_text, labelsize=8)
    ax3.set_xlabel("Number of records", color=chart_text, fontsize=9)
    for i, v in enumerate(vals):
        ax3.text(v + 1, i, str(v), va="center", fontsize=9, fontweight="bold", color=title_color)
    for spine in ax3.spines.values():
        spine.set_color(chart_spine)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# -------------------- HOW IT WORKS --------------------
st.markdown("---")
st.markdown('<p class="section-title">How It Works (CBSE)</p>', unsafe_allow_html=True)
st.markdown("""
1. Dataset is loaded using **Pandas**  
2. Temperature, Humidity and Rainfall are selected as **features**  
3. Data is split into training (80%) and testing (20%) sets  
4. A **Decision Tree** model is trained with **Scikit-learn**  
5. User enters environmental values and clicks Predict  
6. Model predicts crop health class  
7. **Matplotlib** shows the graphs  
""")

st.markdown("---")
st.caption("CropGuard · CBSE Class 12 Computer Science Project · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit")
