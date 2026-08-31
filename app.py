"""
CropGuard - Crop Health Prediction System
CBSE Class 12 | Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="CropGuard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "pred_inputs" not in st.session_state:
    st.session_state.pred_inputs = None

# Load model
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_health_data.csv")
    df = pd.read_csv(path).dropna()
    if len(df.columns) == 4:
        df.columns = ["Humidity", "Temperature", "Rainfall", "Crop_Health"]
    X = df[["Humidity", "Temperature", "Rainfall"]]
    y = df["Crop_Health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, df

model, accuracy, df = load_model()
dark = st.session_state.theme == "dark"

# Theme-aware colors for charts and custom HTML only
if dark:
    page_bg = "#111814"
    card_bg = "#1A221C"
    text_c = "#E8EDE9"
    muted_c = "#A0B0A4"
    primary = "#3DDC84"
    border = "#2A352C"
    chart_bg = "#1A221C"
    chart_text = "#C8D4CC"
    chart_spine = "#3A453C"
    sc = {"Healthy": "#3DDC84", "Moderately Healthy": "#F9A825", "Poor Health": "#EF5350"}
    bar_c = ["#2E7D4F", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg, #0A1F14, #1B5E3B)"
else:
    page_bg = "#F5F7F4"
    card_bg = "#FFFFFF"
    text_c = "#0D3B2E"
    muted_c = "#5A6B5E"
    primary = "#1B5E3B"
    border = "#D0DCD0"
    chart_bg = "#FFFFFF"
    chart_text = "#333333"
    chart_spine = "#CCCCCC"
    sc = {"Healthy": "#2E7D4F", "Moderately Healthy": "#F9A825", "Poor Health": "#C62828"}
    bar_c = ["#1B5E3B", "#43A047", "#F9A825"]
    hero = "linear-gradient(135deg, #0D3B2E, #1B5E3B)"

# Minimal CSS - do NOT override main markdown text colors
st.markdown(f"""
<style>
.stApp {{ background-color: {page_bg}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1100px; }}

/* Sidebar - make it clearly visible */
section[data-testid="stSidebar"] {{
    background-color: {card_bg} !important;
    border-right: 3px solid {primary} !important;
    min-width: 280px !important;
}}
section[data-testid="stSidebar"] > div {{
    background-color: {card_bg} !important;
}}

.stButton > button {{
    background-color: {primary} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🌱 CropGuard")
    st.caption("Crop Health Prediction System")

    if st.button("Switch to " + ("Light Mode ☀️" if dark else "Dark Mode 🌙")):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()

    st.divider()
    st.subheader("Environmental Inputs")
    st.write("Enter the three values below:")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=55.0, value=28.0, step=0.5,
        help="Example: 28"
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0, value=65.0, step=1.0,
        help="Example: 65"
    )
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=400.0, value=120.0, step=5.0,
        help="Example: 120"
    )

    st.write("")
    predict_clicked = st.button("🌿 Predict Crop Health", use_container_width=True)

    if predict_clicked:
        inp = pd.DataFrame(
            [[humidity, temperature, rainfall]],
            columns=["Humidity", "Temperature", "Rainfall"]
        )
        st.session_state.prediction = model.predict(inp)[0]
        st.session_state.pred_inputs = (temperature, humidity, rainfall)

    st.divider()
    st.subheader("Model Info")
    st.write(f"**Records:** {len(df)}")
    st.write(f"**Features:** 3 (Temp, Humidity, Rainfall)")
    st.write(f"**Accuracy:** {accuracy*100:.1f}%")
    st.write("**Model:** Decision Tree")
    st.caption("max_depth = 6")

# ========== MAIN PAGE ==========
st.markdown(f"""
<div style="background:{hero};border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.2rem;">
  <h1 style="color:white;margin:0;font-size:1.8rem;">🌱 CropGuard</h1>
  <p style="color:#D0E8D8;margin:0.3rem 0 0 0;font-size:0.95rem;">
    Predict crop health using temperature, humidity and rainfall
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Crop Health Status")

    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        t, h, r = st.session_state.pred_inputs

        if pred == "Healthy":
            st.success(f"**● HEALTHY**\n\nEnvironmental conditions are favorable for crop growth.")
            st.info("**Recommendation:** Continue regular irrigation and crop monitoring.")
        elif pred == "Moderately Healthy":
            st.warning(f"**● MODERATELY HEALTHY**\n\nConditions are somewhat suitable. Monitor closely.")
            st.info("**Recommendation:** Check soil moisture and temperature regularly.")
        else:
            st.error(f"**● POOR HEALTH**\n\nConditions may be unfavourable for healthy growth.")
            st.info("**Recommendation:** Review irrigation, temperature stress and rainfall.")

        st.write(f"**Your inputs:** Temperature = {t}°C  |  Humidity = {h}%  |  Rainfall = {r} mm")

        st.subheader("Current Conditions")
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))
        fig1.patch.set_facecolor(chart_bg)
        ax1.set_facecolor(chart_bg)
        vals = [t, h, r]
        bars = ax1.bar(["Temp (°C)", "Humidity (%)", "Rainfall (mm)"], vals, color=bar_c, width=0.5)
        ax1.set_ylim(0, max(vals) * 1.35 + 5)
        ax1.tick_params(colors=chart_text, labelsize=9)
        ax1.set_ylabel("Value", color=chart_text)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.03,
                     str(v), ha="center", fontweight="bold", color=primary, fontsize=10)
        for s in ax1.spines.values():
            s.set_color(chart_spine)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    else:
        st.info("👉 Open the **sidebar on the left**, enter Temperature, Humidity and Rainfall, then click **Predict Crop Health**.")

with col2:
    st.subheader("Dataset: Temperature vs Humidity")
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    fig2.patch.set_facecolor(chart_bg)
    ax2.set_facecolor(chart_bg)
    for lab, col in sc.items():
        sub = df[df["Crop_Health"] == lab]
        ax2.scatter(sub["Temperature"], sub["Humidity"], c=col, label=lab, alpha=0.8, s=28, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Temperature (°C)", color=chart_text, fontsize=9)
    ax2.set_ylabel("Humidity (%)", color=chart_text, fontsize=9)
    ax2.tick_params(colors=chart_text, labelsize=8)
    leg = ax2.legend(fontsize=8, loc="best")
    leg.get_frame().set_facecolor(chart_bg)
    for t_ in leg.get_texts():
        t_.set_color(chart_text)
    for s in ax2.spines.values():
        s.set_color(chart_spine)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.subheader("Class Distribution")
    counts = df["Crop_Health"].value_counts()
    order = ["Healthy", "Moderately Healthy", "Poor Health"]
    vals = [int(counts.get(o, 0)) for o in order]
    cols = [sc[o] for o in order]
    fig3, ax3 = plt.subplots(figsize=(5.2, 2.3))
    fig3.patch.set_facecolor(chart_bg)
    ax3.set_facecolor(chart_bg)
    ax3.barh(order, vals, color=cols, height=0.55)
    ax3.tick_params(colors=chart_text, labelsize=8)
    ax3.set_xlabel("Records", color=chart_text, fontsize=9)
    for i, v in enumerate(vals):
        ax3.text(v + 1.5, i, str(v), va="center", fontweight="bold", color=primary, fontsize=9)
    for s in ax3.spines.values():
        s.set_color(chart_spine)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

st.divider()
st.subheader("How It Works (CBSE)")
st.markdown("""
1. The dataset is loaded using **Pandas**
2. Temperature, humidity and rainfall are selected as **features**
3. Data is split into **training (80%)** and **testing (20%)** sets
4. A **Decision Tree** model is trained with **Scikit-learn**
5. User enters environmental values and clicks **Predict**
6. The model predicts the crop health class
7. **Matplotlib** displays the graphs
""")

st.caption("CropGuard · CBSE Class 12 Computer Science · Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit")
