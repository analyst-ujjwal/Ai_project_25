import streamlit as st
import pandas as pd
from src.model_utils import load_model
from src.groq_client import run_on_groq

st.title("Price Prediction (Groq Accelerated)")

uploaded = st.file_uploader("Upload CSV")

model_path = "models/model.joblib"

if uploaded:
    df = pd.read_csv(uploaded)
    st.write(df)

    artifact = load_model(model_path)
    model = artifact["model"]
    pre = artifact["preprocessor"]

    X = df.drop(columns=["price"]) if "price" in df.columns else df
    X_scaled = pre.transform(X)

    try:
        resp = run_on_groq({"inputs": X_scaled.tolist()})
        preds = resp["predictions"]
    except Exception:
        preds = model.predict(X_scaled)

    df["predicted_price"] = preds
    st.write(df)
    st.download_button("Download", df.to_csv(index=False), "predictions.csv")
