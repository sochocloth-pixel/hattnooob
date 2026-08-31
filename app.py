"""
CropGuard – Crop Health Prediction
Clean Figma-style UI | Light & Dark mode
Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="CropGuard", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

# ---- session ----
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "pred_inputs" not in st.session_state:
    st.session_state.pred_inputs = None

# ---- model ----
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    df = pd.read_csv(path).dropna()
    if len(df.columns) == 4:
        df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = DecisionTreeClassifier(max_depth=6, random_state=42)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    return m, acc, df

model, accuracy, df = load_model()
dark = st.session_state.theme == "dark"

# ---- theme colours ----
if dark:
    BG, CARD, BORDER = "#0F1410", "#1A221C", "#2A352C"
    TEXT, MUTED, PRIMARY = "#E8EDE9", "#9AAB9E", "#3DDC84"
    HERO1, HERO2 = "#0A1F14", "#1B5E3B"
    OK_BG, OK_BD, OK_TX = "#143D2A", "#3DDC84", "#3DDC84"
    MID_BG, MID_BD, MID_TX = "#3D3010", "#F9A825", "#F9A825"
    BAD_BG, BAD_BD, BAD_TX = "#3D1A1A", "#EF5350", "#EF5350"
    INFO_BG, CHIP_BG, CHIP_TX = "#1A221C", "#243028", "#3DDC84"
    C_BG, C_TXT, C_SP = "#1A221C", "#C8D4CC", "#3A453C"
    SC = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    BAR_C = ["#2E7D4F", "#43A047", "#F9A825"]
else:
    BG, CARD, BORDER = "#F4F6F3", "#FFFFFF", "#D8E0D8"
    TEXT, MUTED, PRIMARY = "#0D3B2E", "#5A6B5E", "#1B5E3B"
    HERO1, HERO2 = "#0D3B2E", "#1B5E3B"
    OK_BG, OK_BD, OK_TX = "#E8F5E9", "#2E7D4F", "#1B5E3B"
    MID_BG, MID_BD, MID_TX = "#FFF8E1", "#F9A825", "#E65100"
    BAD_BG, BAD_BD, BAD_TX = "#FFEBEE", "#C62828", "#C62828"
    INFO_BG, CHIP_BG, CHIP_TX = "#FFFFFF", "#E8F0E8", "#1B5E3B"
    C_BG, C_TXT, C_SP = "#FFFFFF", "#333333", "#CCCCCC"
    SC = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    BAR_C = ["#1B5E3B", "#43A047", "#F9A825"]

# ---- CSS ----
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background: {BG} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.25rem !important; max-width: 1120px; }}

section[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {{
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] label {{
    font-weight: 600 !important; font-size: 0.82rem !important; color: {MUTED} !important;
}}

.stButton > button {{
    background: {PRIMARY} !important; color: {"#0D3B2E" if dark else "#FFFFFF"} !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; width: 100%;
}}
.stNumberInput input {{
    border-radius: 8px !important; border: 1.5px solid {BORDER} !important;
    background: {CARD} !important; color: {TEXT} !important;
}}
div[data-testid="stMetricValue"] {{ color: {PRIMARY} !important; font-weight: 800 !important; }}
div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    tcol, bcol = st.columns([4, 1])
    with tcol:
        st.markdown(f"<h3 style='color:{TEXT};margin:0;'>🌱 CropGuard</h3>", unsafe_allow_html=True)
    with bcol:
        if st.button("🌙" if not dark else "☀️"):
            st.session_state.theme = "dark" if not dark else "light"
            st.rerun()

    st.markdown(f"<p style='color:{MUTED};font-size:0.8rem;margin:0 0 1rem 0;'>Crop Health Prediction</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='color:{TEXT};font-weight:700;margin-bottom:0.5rem;'>Environmental Conditions</p>", unsafe_allow_html=True)

    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=120.0, step=5.0)

    st.write("")
    if st.button("🌿  Predict Crop Health"):
        inp = pd.DataFrame([[humidity, temperature, rainfall]],
                           columns=["Humidity", "Temperature", "Rainfall"])
        st.session_state.prediction = model.predict(inp)[0]
        st.session_state.pred_inputs = (temperature, humidity, rainfall)

    st.markdown("---")
    st.markdown(f"<p style='color:{TEXT};font-weight:700;'>Model Info</p>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Records", len(df))
    b.metric("Features", 3)
    c.metric("Accuracy", f"{accuracy*100:.0f}%")
    st.caption("Decision Tree · max_depth = 6")

# ==================== HERO ====================
st.markdown(f"""
<div style="background:linear-gradient(135deg,{HERO1},{HERO2});border-radius:16px;
padding:1.6rem 2rem;margin-bottom:1.4rem;color:#fff;">
  <div style="font-size:1.85rem;font-weight:800;letter-spacing:-0.4px;">🌱 CropGuard</div>
  <div style="opacity:0.9;margin-top:0.3rem;font-size:0.95rem;">
    Predict crop health from temperature, humidity &amp; rainfall
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== MAIN ====================
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:1.05rem;margin-bottom:0.7rem;'>Crop Health Status</p>", unsafe_allow_html=True)

    if st.session_state.prediction:
        pred = st.session_state.prediction
        t, h, r = st.session_state.pred_inputs

        if pred == "Healthy":
            bg, bd, tx, desc, rec = OK_BG, OK_BD, OK_TX, \
                "Environmental conditions are favorable for crop growth.", \
                "Continue regular irrigation and crop monitoring."
        elif pred == "Moderately Healthy":
            bg, bd, tx, desc, rec = MID_BG, MID_BD, MID_TX, \
                "Conditions are somewhat suitable. Monitor closely.", \
                "Check soil moisture and temperature regularly."
        else:
            bg, bd, tx, desc, rec = BAD_BG, BAD_BD, BAD_TX, \
                "Conditions may be unfavourable for healthy growth.", \
                "Review irrigation, temperature stress and rainfall."

        st.markdown(f"""
        <div style="background:{bg};border:2px solid {bd};border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:0.8rem;">
          <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{MUTED};">Result</div>
          <div style="font-size:1.55rem;font-weight:800;color:{tx};margin:0.15rem 0 0.35rem;">● {pred.upper()}</div>
          <div style="color:{TEXT};font-size:0.9rem;">{desc}</div>
        </div>
        <div style="margin-bottom:0.8rem;">
          <span style="background:{CHIP_BG};color:{CHIP_TX};border-radius:20px;padding:0.28rem 0.8rem;
          font-size:0.82rem;font-weight:600;margin-right:0.3rem;display:inline-block;">🌡 {t} °C</span>
          <span style="background:{CHIP_BG};color:{CHIP_TX};border-radius:20px;padding:0.28rem 0.8rem;
          font-size:0.82rem;font-weight:600;margin-right:0.3rem;display:inline-block;">💧 {h} %</span>
          <span style="background:{CHIP_BG};color:{CHIP_TX};border-radius:20px;padding:0.28rem 0.8rem;
          font-size:0.82rem;font-weight:600;display:inline-block;">🌧 {r} mm</span>
        </div>
        <div style="background:{INFO_BG};border:1px solid {BORDER};border-left:4px solid {PRIMARY};
        border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:1rem;">
          <div style="font-size:0.7rem;font-weight:700;color:{PRIMARY};text-transform:uppercase;letter-spacing:0.8px;">Recommendation</div>
          <div style="color:{TEXT};font-size:0.9rem;margin-top:0.2rem;">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:1rem;margin:0.5rem 0 0.5rem;'>Current Conditions</p>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.4, 2.6))
        fig1.patch.set_facecolor(C_BG)
        ax1.set_facecolor(C_BG)
        vals = [t, h, r]
        bars = ax1.bar(["Temp (°C)", "Humidity (%)", "Rainfall (mm)"], vals, color=BAR_C, width=0.5)
        ax1.set_ylim(0, max(vals) * 1.35 + 5)
        ax1.tick_params(colors=C_TXT, labelsize=8)
        ax1.set_ylabel("Value", color=C_TXT, fontsize=9)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                     str(v), ha="center", fontsize=9, fontweight="bold", color=PRIMARY)
        for s in ax1.spines.values():
            s.set_color(C_SP)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    else:
        st.markdown(f"""
        <div style="background:{CARD};border:2px dashed {BORDER};border-radius:14px;
        padding:2.2rem 1.5rem;text-align:center;">
          <div style="font-size:2.2rem;margin-bottom:0.4rem;">🌿</div>
          <div style="color:{TEXT};font-weight:700;font-size:1rem;">Ready to predict</div>
          <div style="color:{MUTED};font-size:0.88rem;margin-top:0.3rem;line-height:1.45;">
            Enter Temperature, Humidity and Rainfall<br>in the sidebar, then click <b>Predict</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:1.05rem;margin-bottom:0.7rem;'>Dataset Overview</p>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(5.1, 3.3))
    fig2.patch.set_facecolor(C_BG)
    ax2.set_facecolor(C_BG)
    for lab, col in SC.items():
        sub = df[df["Crop_Health"] == lab]
        ax2.scatter(sub["Temperature"], sub["Humidity"], c=col, label=lab, alpha=0.8, s=28, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Temperature (°C)", color=C_TXT, fontsize=9)
    ax2.set_ylabel("Humidity (%)", color=C_TXT, fontsize=9)
    ax2.tick_params(colors=C_TXT, labelsize=8)
    leg = ax2.legend(fontsize=7, loc="best", framealpha=0.95)
    leg.get_frame().set_facecolor(C_BG)
    leg.get_frame().set_edgecolor(C_SP)
    for t_ in leg.get_texts():
        t_.set_color(C_TXT)
    for s in ax2.spines.values():
        s.set_color(C_SP)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Temperature vs Humidity", color=PRIMARY, fontsize=11, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown(f"<p style='color:{TEXT};font-weight:700;font-size:1rem;margin:0.6rem 0 0.4rem;'>Class Distribution</p>", unsafe_allow_html=True)
    counts = df["Crop_Health"].value_counts()
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [int(counts.get(o, 0)) for o in order]
    cols = [SC[o] for o in order]

    fig3, ax3 = plt.subplots(figsize=(5.1, 2.3))
    fig3.patch.set_facecolor(C_BG)
    ax3.set_facecolor(C_BG)
    ax3.barh(order, vals, color=cols, height=0.55)
    ax3.tick_params(colors=C_TXT, labelsize=8)
    ax3.set_xlabel("Records", color=C_TXT, fontsize=9)
    for i, v in enumerate(vals):
        ax3.text(v + 1.5, i, str(v), va="center", fontsize=9, fontweight="bold", color=PRIMARY)
    for s in ax3.spines.values():
        s.set_color(C_SP)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ==================== HOW IT WORKS ====================
st.markdown("---")
st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:1.3rem 1.5rem;margin-bottom:1rem;">
  <div style="color:{TEXT};font-weight:700;font-size:1.05rem;margin-bottom:0.9rem;">How It Works (CBSE)</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem 1.5rem;">
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">1.</b> Load dataset with <b>Pandas</b></div>
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">2.</b> Select features: Temp, Humidity, Rainfall</div>
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">3.</b> Split into train (80%) &amp; test (20%)</div>
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">4.</b> Train <b>Decision Tree</b> (Scikit-learn)</div>
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">5.</b> User enters values → Predict</div>
    <div style="color:{TEXT};font-size:0.88rem;line-height:1.5;"><b style="color:{PRIMARY};">6.</b> Show result + <b>Matplotlib</b> graphs</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center;color:{MUTED};font-size:0.78rem;padding:0.5rem 0 1rem;">
  <b style="color:{PRIMARY};">CropGuard</b> · CBSE Class 12 · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
