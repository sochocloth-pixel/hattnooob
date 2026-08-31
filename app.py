"""
CropGuard – AI-Based Crop Health Prediction System
Public Web Version (Streamlit)

Same ML backend as the CBSE Tkinter project:
Pandas + NumPy + Scikit-learn Decision Tree + Matplotlib
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
    page_title="CropGuard – Crop Health Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS (Nature theme) --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

    .stApp {
        background-color: #F7F5F0;
    }
    .main-header {
        background: linear-gradient(135deg, #1B5E3B 0%, #2E7D4F 100%);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }
    .result-healthy {
        background: #E8F5E9;
        border-left: 6px solid #2E7D4F;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-moderate {
        background: #FFF8E1;
        border-left: 6px solid #E8A838;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-poor {
        background: #FFEBEE;
        border-left: 6px solid #C62828;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #E0DDD5;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #5A5A5A;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD & TRAIN MODEL (cached) --------------------
@st.cache_resource
def load_and_train_model():
    """
    Load CSV, train DecisionTreeClassifier, return model + accuracy + dataframe.
    Cached so training happens only once.
    """
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


# -------------------- HEADER --------------------
st.markdown("""
<div class="main-header">
    <h1>🌱 CropGuard</h1>
    <p>AI-Based Crop Health Prediction System</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Predict crop health using **temperature**, **humidity** and **rainfall** conditions. "
    "Powered by a Decision Tree model trained with Scikit-learn."
)


# -------------------- SIDEBAR – INPUTS --------------------
with st.sidebar:
    st.header("🌤 Environmental Conditions")
    st.caption("Enter realistic values and click Predict")

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=55.0,
        value=28.0, step=0.5,
        help="Typical crop range: 15–40 °C"
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0,
        value=65.0, step=1.0,
        help="Relative humidity between 0 and 100"
    )
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=400.0,
        value=120.0, step=5.0,
        help="Rainfall amount in millimetres"
    )

    predict_btn = st.button("🌿 Predict Crop Health", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📁 Dataset & Model")
    st.metric("Total Records", len(df))
    st.metric("Features", 3)
    st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
    st.caption("DecisionTreeClassifier • max_depth=6")


# -------------------- MAIN AREA --------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("📊 Crop Health Status")

    if predict_btn:
        # Prepare input as DataFrame (keeps feature names)
        input_df = pd.DataFrame(
            [[humidity, temperature, rainfall]],
            columns=["Humidity", "Temperature", "Rainfall"]
        )
        prediction = model.predict(input_df)[0]

        if prediction == "Healthy":
            st.markdown(f"""
            <div class="result-healthy">
                <h2 style="color:#1B5E3B; margin:0;">● HEALTHY</h2>
                <p style="margin:0.5rem 0 0 0;">The current environmental conditions are favorable for crop growth.</p>
            </div>
            """, unsafe_allow_html=True)
            rec = "Conditions appear favourable. Continue regular irrigation and crop monitoring."
            rec_color = "#E8F5E9"
        elif prediction == "Moderately Healthy":
            st.markdown(f"""
            <div class="result-moderate">
                <h2 style="color:#E65100; margin:0;">● MODERATELY HEALTHY</h2>
                <p style="margin:0.5rem 0 0 0;">Environmental conditions are somewhat suitable for the crop.</p>
            </div>
            """, unsafe_allow_html=True)
            rec = "Monitor soil moisture and temperature regularly. Adjust irrigation if needed."
            rec_color = "#FFF8E1"
        else:
            st.markdown(f"""
            <div class="result-poor">
                <h2 style="color:#C62828; margin:0;">● POOR HEALTH</h2>
                <p style="margin:0.5rem 0 0 0;">Conditions may be unfavourable for healthy crop growth.</p>
            </div>
            """, unsafe_allow_html=True)
            rec = "Check irrigation, temperature stress and rainfall conditions carefully."
            rec_color = "#FFEBEE"

        st.markdown(f"""
        **Your inputs**  
        Temperature: **{temperature} °C** &nbsp;|&nbsp;  
        Humidity: **{humidity} %** &nbsp;|&nbsp;  
        Rainfall: **{rainfall} mm**
        """)

        st.info(f"💡 **Recommendation:** {rec}")

        # ----- Bar chart of current conditions -----
        st.subheader("📈 Current Environmental Conditions")
        fig1, ax1 = plt.subplots(figsize=(6, 3.2))
        labels = ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)"]
        values = [temperature, humidity, rainfall]
        colors = ["#1B5E3B", "#2E7D4F", "#E8A838"]
        bars = ax1.bar(labels, values, color=colors, width=0.55, edgecolor="white")
        ax1.set_ylabel("Value")
        ax1.set_title("Current Inputs", fontweight="bold", color="#1B5E3B")
        ax1.set_ylim(0, max(values) * 1.25 + 5)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{val}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    else:
        st.info("Enter values in the sidebar and click **🌿 Predict Crop Health** to see the result.")


with col2:
    st.subheader("📉 Dataset: Temperature vs Humidity")
    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    colour_map = {
        "Healthy": "#2E7D4F",
        "Moderately Healthy": "#E8A838",
        "Poor Health": "#C62828"
    }
    for label, colour in colour_map.items():
        subset = df[df["Crop_Health"] == label]
        ax2.scatter(
            subset["Temperature"], subset["Humidity"],
            c=colour, label=label, alpha=0.75, s=32,
            edgecolors="white", linewidths=0.4
        )
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Humidity (%)")
    ax2.set_title("Training Data Distribution", fontweight="bold", color="#1B5E3B")
    ax2.legend(fontsize=8, loc="best")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Class distribution
    st.subheader("Class Distribution in Dataset")
    class_counts = df["Crop_Health"].value_counts()
    st.bar_chart(class_counts)


# -------------------- HOW IT WORKS --------------------
with st.expander("📘 How It Works (for students / viva)", expanded=False):
    st.markdown("""
    1. The dataset is loaded using **Pandas**.
    2. Temperature, humidity and rainfall are selected as **features**.
    3. The data is divided into **training** (80%) and **testing** (20%) sets.
    4. A **Decision Tree** model is trained using **Scikit-learn**.
    5. The user enters environmental conditions on this website.
    6. The model predicts the crop health class.
    7. **Matplotlib** displays the environmental data visually.

    **Why Decision Tree?**  
    It is easy to understand (works like a flowchart of if-else rules),  
    suitable for classification, and appropriate for Class 12 level.
    """)


# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
    CropGuard • CBSE Class 12 Computer Science inspired project<br>
    Python • Pandas • NumPy • Matplotlib • Scikit-learn • Streamlit
</div>
""", unsafe_allow_html=True)
