from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_abroad_10k.csv"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    df["Study_Destination"] = df["Study_Destination"].map({"India": 0, "Abroad": 1})
    drop_cols = [c for c in ["index"] if c in df.columns]
    X = df.drop(columns=["Study_Destination"] + drop_cols)
    y = df["Study_Destination"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def evaluate_model(name: str, model, preprocessor: ColumnTransformer, X: pd.DataFrame, y) -> dict:
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {"name": name, "accuracy": acc.mean(), "roc_auc": auc.mean()}


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_xy(df)
    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print("MODEL COMPARISON (5-Fold Stratified CV)")
    print(f"{'Model':<22} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 44)

    results = []
    for name, model in models.items():
        r = evaluate_model(name, model, preprocessor, X, y)
        results.append(r)
        print(f"{r['name']:<22} {r['accuracy']:>10.4f} {r['roc_auc']:>10.4f}")

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nBest model: {best['name']} (ROC-AUC: {best['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
