"""
Stress-tests the Random Forest model on borderline students:
low CGPA (<= 7.0), low budget (<= 10 LPA), low family income (<= 10 LPA).
Useful for checking whether the model generalises beyond obvious high-achievers.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_abroad_10k.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df["Study_Destination"] = df["Study_Destination"].map({"India": 0, "Abroad": 1})

    # Keep only borderline / hard cases
    df = df[
        (df["CGPA"] <= 7.0)
        & (df["Budget_LPA"] <= 10)
        & (df["Family_Income_LPA"] <= 10)
    ]

    print(f"Stress-test subset size: {df.shape[0]} rows")
    if df.shape[0] < 10:
        print("Too few samples for cross-validation — skipping.")
        return

    X = df.drop(columns=["Study_Destination"] + [c for c in ["index"] if c in df.columns])
    y = df["Study_Destination"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print("\nSTRESS TEST RESULTS (borderline students)")
    print(f"Accuracy : {acc.mean():.4f} ± {acc.std():.4f}")
    print(f"ROC-AUC  : {auc.mean():.4f} ± {auc.std():.4f}")


if __name__ == "__main__":
    main()
