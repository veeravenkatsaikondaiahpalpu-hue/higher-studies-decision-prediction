"""
Train model for predicting Study Destination (India vs Abroad)
based on academic + financial features.

Outputs:
- Saved model (.pkl)
- Saved scaler (.pkl)
- Saved label encoders (.pkl)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ======================
# Paths
# ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_abroad_dataset_with_india.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ======================
# Load dataset
# ======================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    return df


# ======================
# Preprocess
# ======================
def preprocess(df: pd.DataFrame):
    """
    Encodes categorical columns and scales numeric features.
    Returns X, y, scaler, encoders.
    """
    df = df.copy()

    target_col = "Study_Destination"

    # --- Drop columns that are not helpful (if present) ---
    drop_cols = ["index"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- Encode target ---
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

    # --- Encode categorical features if present ---
    feature_encoders = {}

    cat_cols = []
    for col in ["Field_of_Study", "Country"]:
        if col in df.columns:
            cat_cols.append(col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        feature_encoders[col] = le

    # --- Define features ---
    feature_cols = [
        "CGPA",
        "GRE_Score",
        "TOEFL_Score",
        "Family_Income_LPA",
        "Budget_LPA",
        "Research_Experience",
        "Scholarship",
    ]

    # Only keep available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    # --- Scale numeric columns ---
    numeric_cols = ["CGPA", "GRE_Score", "TOEFL_Score", "Family_Income_LPA", "Budget_LPA"]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    encoders = {
        "target_encoder": target_encoder,
        "feature_encoders": feature_encoders,
        "feature_cols": feature_cols
    }

    return X, y, scaler, encoders


# ======================
# Train + Evaluate
# ======================
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print(f"✅ Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        results[name] = {
            "model": model,
            "accuracy": acc
        }

    return results


def save_artifacts(best_model_name, best_model, scaler, encoders):
    joblib.dump(best_model, MODEL_DIR / f"{best_model_name}.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    print("\n✅ Saved artifacts to:", MODEL_DIR)


def main():
    print("📌 Loading dataset:", DATA_PATH)
    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    X, y, scaler, encoders = preprocess(df)
    results = train_models(X, y)

    # Select best model by accuracy
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_model_name]["model"]

    print("\n🏆 Best model:", best_model_name, "| Accuracy:", results[best_model_name]["accuracy"])
    save_artifacts(best_model_name, best_model, scaler, encoders)


if __name__ == "__main__":
    main()
