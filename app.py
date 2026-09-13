"""
Medicine Stockout Prediction - Streamlit Demo App

Run locally with:
    streamlit run app.py

Or deploy for free at https://share.streamlit.io by connecting this GitHub repo.

This app loads a model trained in Week3_Medicine_Stockout_Prediction.ipynb
(saved with joblib as xgb_stockout_model.pkl) and lets a user enter one
product/facility's monthly figures to get a stockout-risk prediction.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medicine Stockout Predictor", page_icon="💊", layout="centered")

# --- Load the trained model -------------------------------------------------
# This file is created in the notebook with:
#   import joblib
#   joblib.dump(best_xgb, "xgb_stockout_model.pkl")
MODEL_PATH = "xgb_stockout_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# --- Feature lists (must match the notebook exactly) ------------------------
FEATURE_COLS = [
    "openBalance", "received", "consumption", "closeBalance",
    "Days_of_Stock_Remaining",
    "Previous_Month_Consumption", "Consumption_Change", "Rolling_3_Month_Consumption",
    "Previous_Month_Received", "Rolling_3_Month_Received",
    "Previous_Month_OpenBalance", "Previous_Month_CloseBalance",
    "Previous_Month_Stockout",
]

# Same columns the notebook applies log1p to - must match Section "Handling
# skewness with log1p" exactly, or predictions will be wrong.
SKEWED_COLS = [
    "openBalance", "received", "consumption", "closeBalance",
    "Days_of_Stock_Remaining",
    "Previous_Month_Consumption", "Rolling_3_Month_Consumption",
    "Previous_Month_Received", "Rolling_3_Month_Received",
    "Previous_Month_OpenBalance", "Previous_Month_CloseBalance",
]

# --- UI ----------------------------------------------------------------------
st.title("💊 Medicine Stockout Predictor")
st.write(
    "Enter this month's figures for one product at one facility to estimate "
    "the risk that it will be **out of stock next month**."
)

if not model_loaded:
    st.error(
        f"Model file `{MODEL_PATH}` not found. Train the model in the notebook, "
        f"run `joblib.dump(best_xgb, '{MODEL_PATH}')`, and place the file in the "
        f"same folder as this app."
    )
    st.stop()

st.subheader("This month's figures")
col1, col2 = st.columns(2)
with col1:
    openBalance = st.number_input("Opening balance", min_value=0.0, value=100.0)
    received = st.number_input("Received this month", min_value=0.0, value=0.0)
    consumption = st.number_input("Consumed this month", min_value=0.0, value=50.0)
    closeBalance = st.number_input("Closing balance", min_value=0.0, value=50.0)
with col2:
    days_of_stock = st.number_input("Days of stock remaining", min_value=0.0, value=30.0)
    prev_stockout = st.selectbox("Was there a stockout last month?", ["No", "Yes"])

st.subheader("Recent history")
col3, col4 = st.columns(2)
with col3:
    prev_consumption = st.number_input("Previous month consumption", min_value=0.0, value=45.0)
    consumption_change = st.number_input("Change in consumption vs last month", value=5.0)
    rolling_consumption = st.number_input("3-month average consumption", min_value=0.0, value=48.0)
with col4:
    prev_received = st.number_input("Previous month received", min_value=0.0, value=0.0)
    rolling_received = st.number_input("3-month average received", min_value=0.0, value=20.0)
    prev_open = st.number_input("Previous month opening balance", min_value=0.0, value=90.0)
    prev_close = st.number_input("Previous month closing balance", min_value=0.0, value=100.0)

threshold = st.slider(
    "Decision threshold (lower = catches more real stockouts, but more false alarms)",
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="The Week 3 analysis found 0.3 catches ~82% of real stockouts.",
)

if st.button("Predict stockout risk", type="primary"):
    row = pd.DataFrame([{
        "openBalance": openBalance,
        "received": received,
        "consumption": consumption,
        "closeBalance": closeBalance,
        "Days_of_Stock_Remaining": days_of_stock,
        "Previous_Month_Consumption": prev_consumption,
        "Consumption_Change": consumption_change,
        "Rolling_3_Month_Consumption": rolling_consumption,
        "Previous_Month_Received": prev_received,
        "Rolling_3_Month_Received": rolling_received,
        "Previous_Month_OpenBalance": prev_open,
        "Previous_Month_CloseBalance": prev_close,
        "Previous_Month_Stockout": 1 if prev_stockout == "Yes" else 0,
    }])[FEATURE_COLS]

    # Apply the SAME log1p transform used during training - skipping this
    # step would make predictions meaningless, since the model was trained
    # on log1p-transformed values.
    row[SKEWED_COLS] = np.log1p(row[SKEWED_COLS])

    proba = model.predict_proba(row)[0, 1]
    is_risk = proba >= threshold

    st.divider()
    if is_risk:
        st.error(f"⚠️ High stockout risk — estimated probability: **{proba:.0%}**")
    else:
        st.success(f"✅ Low stockout risk — estimated probability: **{proba:.0%}**")

    st.progress(min(float(proba), 1.0))
    st.caption(
        f"Prediction uses a threshold of {threshold:.2f}: "
        f"any probability at or above this is flagged as high risk."
    )

st.divider()
st.caption(
    "Demo built for the Medicine Stockout Prediction student project (Week 4). "
    "Model: XGBoost tuned via GridSearchCV, ROC-AUC 0.838 on the held-out test set."
)
