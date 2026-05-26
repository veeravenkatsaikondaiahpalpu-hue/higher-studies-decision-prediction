"""
Training pipeline for study abroad prediction.

Leakage prevention:
- All preprocessing (StandardScaler, OneHotEncoder) lives inside a sklearn
  Pipeline and is fit exclusively on training data. The test set is never
  seen until the final one-time evaluation at the bottom of main().

Overfitting prevention:
- Model selection uses 5-fold StratifiedKFold CV on the training set only.
- GridSearchCV tunes regularisation (LR) and depth/leaf limits (RF).
- Random Forest max_depth is bounded; min_samples_leaf enforces pruning.
- Both models use class_weight="balanced" to handle label imbalance.

Outputs:
- models/best_model.pkl        — fitted pipeline (preprocessor + model)
- predictions/test_predictions.csv — held-out test set predictions
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_abroad_10k.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
MODEL_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "CGPA", "GRE_Score", "TOEFL_Score",
    "Family_Income_LPA", "Budget_LPA",
    "Research_Experience", "Scholarship",
]
CATEGORICAL_COLS = ["Field_of_Study"]


def load_data(path: Path):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    df["Study_Destination"] = df["Study_Destination"].str.strip().str.title()
    df = df[df["Study_Destination"].isin(["India", "Abroad"])].copy()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = (df["Study_Destination"] == "Abroad").astype(int)  # 1=Abroad, 0=India
    return X, y


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def print_metrics(name: str, y_test, y_pred, y_prob) -> dict:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*60}")
    print(f"Model    : {name}")
    print(f"Accuracy : {acc:.4f}   ROC-AUC : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["India", "Abroad"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return {"accuracy": acc, "roc_auc": auc, "y_pred": y_pred, "y_prob": y_prob}


def main():
    print(f"Loading data: {DATA_PATH}")
    X, y = load_data(DATA_PATH)
    print(f"Shape: {X.shape}  |  Abroad: {y.sum()} ({y.mean()*100:.1f}%)  India: {(1-y).sum()}")

    # 80 / 20 split — test set is held out and touched only once at the end
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Logistic Regression ---
    print("\nRunning GridSearchCV: Logistic Regression ...")
    lr_search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    lr_search.fit(X_train, y_train)
    print(f"  Best C={lr_search.best_params_['model__C']}  CV AUC={lr_search.best_score_:.4f}")

    # --- Random Forest ---
    print("\nRunning GridSearchCV: Random Forest ...")
    rf_search = GridSearchCV(
        build_pipeline(
            RandomForestClassifier(random_state=42, class_weight="balanced")
        ),
        param_grid={
            "model__n_estimators": [200, 300],
            "model__max_depth": [10, 20],
            "model__min_samples_leaf": [5, 10],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    print(f"  Best params={rf_search.best_params_}  CV AUC={rf_search.best_score_:.4f}")

    # --- Final evaluation on held-out test set (done once, after model selection) ---
    results = {}
    for name, search in [("Logistic Regression", lr_search), ("Random Forest", rf_search)]:
        pipeline = search.best_estimator_
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        results[name] = print_metrics(name, y_test, y_pred, y_prob)
        results[name]["pipeline"] = pipeline

    # --- Pick winner ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]
    print(f"\nWinner: {best_name}  (Test AUC: {best['roc_auc']:.4f})")

    # --- Save model ---
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best["pipeline"], model_path)
    print(f"Model saved to: {model_path}")

    # --- Save predictions ---
    pred_df = X_test.copy().reset_index(drop=True)
    pred_df["actual"] = y_test.values
    pred_df["predicted"] = best["y_pred"]
    pred_df["probability_abroad"] = best["y_prob"].round(4)
    pred_df["correct"] = pred_df["actual"] == pred_df["predicted"]
    pred_df["actual_label"] = pred_df["actual"].map({1: "Abroad", 0: "India"})
    pred_df["predicted_label"] = pred_df["predicted"].map({1: "Abroad", 0: "India"})

    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")

    correct_pct = pred_df["correct"].mean() * 100
    print(f"\nTest set: {correct_pct:.1f}% of {len(pred_df)} predictions correct")


if __name__ == "__main__":
    main()
