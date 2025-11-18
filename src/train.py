import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from model_utils import load_csv, split_features_target, Preprocessor, save_model

def main():
    models_dir = "models"
    data_path = "data/sample_data.csv"
    model_path = os.path.join(models_dir, "model.joblib")

    # 1. Create models/ folder if missing
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created folder: {models_dir}")

    # 2. Load default dataset (no arguments needed)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    print(f"Using dataset: {data_path}")

    df = load_csv(data_path)
    X, y = split_features_target(df)

    # 3. Preprocess
    pre = Preprocessor()
    X_scaled = pre.fit_transform(X)

    # 4. Train model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 5. Evaluate
    preds = model.predict(X_scaled)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    print("Training completed.")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 6. Save model & preprocessor
    save_model({"model": model, "preprocessor": pre}, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
