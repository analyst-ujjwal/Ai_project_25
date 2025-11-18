import argparse
import pandas as pd
from model_utils import load_csv, load_model
from groq_client import run_on_groq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    df = load_csv(args.input)
    X = df.drop(columns=["price"]) if "price" in df.columns else df

    artifact = load_model(args.model)
    model = artifact["model"]
    pre = artifact["preprocessor"]

    X_scaled = pre.transform(X)

    try:
        resp = run_on_groq({"inputs": X_scaled.tolist()})
        preds = resp["predictions"]
    except Exception:
        preds = model.predict(X_scaled)

    df["predicted_price"] = preds
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
