import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess
from src.synthetic_data import generate_synthetic_data
from src.model import train_model
from src.threshold import adaptive_threshold
from src.evaluation import evaluate

st.set_page_config(
    page_title="State-wise Groundwater Prediction",
    layout="wide"
)

st.title("💧 State-wise Groundwater Level Prediction")

@st.cache_data
def load_data():
    return load_and_preprocess(
        "data/Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv"
    )

df = load_data()

# -----------------------------
# STATE SELECTION
# -----------------------------
st.sidebar.header("📍 Select State")
states = sorted(df["state"].unique())
selected_state = st.sidebar.selectbox("State", states)

df_state = df[df["state"] == selected_state]

st.subheader(f"Groundwater Prediction for {selected_state}")
st.dataframe(df_state.head())

# -----------------------------
# CONFIGURATION
# -----------------------------
st.sidebar.header("⚙️ Configuration")

synthetic_samples = st.sidebar.slider("Synthetic samples", 200, 1500, 800, 100)
noise_ratio = st.sidebar.slider("Noise ratio", 0.01, 0.1, 0.05, 0.01)
threshold_q = st.sidebar.slider("Threshold quantile", 0.05, 0.3, 0.15, 0.05)

prepare_btn = st.sidebar.button("Generate Synthetic Data")
train_btn = st.sidebar.button("Train Model")

# -----------------------------
# SYNTHETIC DATA
# -----------------------------
if prepare_btn:
    synthetic = generate_synthetic_data(
        df_state,
        n_samples=synthetic_samples,
        noise_ratio=noise_ratio
    )
    final_df = pd.concat([df_state, synthetic], ignore_index=True)
    st.session_state["final_df"] = final_df
    st.success("Synthetic data generated")

# -----------------------------
# TRAINING
# -----------------------------
if train_btn:
    train_df = st.session_state.get("final_df", df_state)

    model, X_test, y_test = train_model(train_df)
    preds = model.predict(X_test)

    st.session_state["y_test"] = y_test
    st.session_state["preds"] = preds

    st.success("Model trained successfully")

# -----------------------------
# RESULTS
# -----------------------------
if "y_test" in st.session_state:
    y_test = st.session_state["y_test"]
    preds = st.session_state["preds"]

    mae, rmse = evaluate(y_test, preds)
    threshold = adaptive_threshold(y_test.values, preds, threshold_q)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("Adaptive Threshold", f"{threshold:.2f} mbgl")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values, label="Actual")
    ax.plot(preds, label="Predicted", linestyle="--")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.legend()
    ax.set_ylabel("Groundwater Level (mbgl)")
    st.pyplot(fig)
