from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_abroad_10k.csv"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_xy(df):
    target = "Study_Destination"

    # Map target to binary
    df[target] = df[target].map({"India": 0, "Abroad": 1})

    X = df.drop(columns=[target])
    y = df[target]

    # Drop unwanted columns if present
    drop_cols = ["index"]
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    return X, y


def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline


def run_cross_validation(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    print("\n✅ Cross Validation Results (5-Fold Stratified)")
    print(f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"ROC-AUC : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")


def main():
    print(f"Loading dataset: {DATA_PATH}")
    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)
    X, y = prepare_xy(df)

    pipeline = build_pipeline(X)

    run_cross_validation(pipeline, X, y)


if __name__ == "__main__":
    main()
