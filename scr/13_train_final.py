import json
import os
import time

import joblib
import pandas as pd
import polars as pl
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/final_ml_dataset.parquet"
TARGET_COL = "target_win"
DROP_COLS = [TARGET_COL, "match_id"]

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.50

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.joblib")
MODEL_PATH_LEGACY = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "columns.json")
METRICS_PATH = os.path.join(MODEL_DIR, "xgboost_metrics.json")

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 1,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbosity": 0,
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pl.read_parquet(DATA_PATH)
    print(f"Original shape: {df.shape}")

    sample_size = min(SAMPLE_SIZE, df.height)
    if sample_size < df.height:
        df = df.sample(n=sample_size, seed=RANDOM_STATE)
    print(f"Sampled shape: {df.shape}")

    pdf = df.to_pandas()
    for column in pdf.select_dtypes(include=["int64"]).columns:
        pdf[column] = pd.to_numeric(pdf[column], downcast="integer")
    for column in pdf.select_dtypes(include=["float64"]).columns:
        pdf[column] = pd.to_numeric(pdf[column], downcast="float")

    X = pdf.drop(columns=DROP_COLS)
    y = pdf[TARGET_COL]
    print(f"Feature matrix shape: {X.shape}")
    return X, y


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**XGBOOST_PARAMS)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Decision threshold: {THRESHOLD}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_importance = pd.Series(model.feature_importances_, index=X_test.columns)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values(ascending=False).head(10))

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def save_artifacts(model, columns: list[str], metrics: dict, elapsed_seconds: float) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, MODEL_PATH_LEGACY)

    with open(SCHEMA_PATH, "w") as file_handle:
        json.dump(columns, file_handle, indent=2)

    payload = {
        "model_name": "xgboost",
        "sample_size": SAMPLE_SIZE,
        "threshold": THRESHOLD,
        "training_time_seconds": round(float(elapsed_seconds), 1),
        "parameters": XGBOOST_PARAMS,
        **metrics,
    }

    with open(METRICS_PATH, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)

    print("\n=== Saved Artifacts ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Legacy model copy: {MODEL_PATH_LEGACY}")
    print(f"Feature schema: {SCHEMA_PATH}")
    print(f"Metrics: {METRICS_PATH}")


def main():
    print("=== FINAL XGBOOST TRAINING ===")
    started_at = time.time()

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_model()
    model.fit(X_train, y_train)
    print("XGBoost trained")

    metrics = evaluate_model(model, X_test, y_test)
    elapsed_seconds = time.time() - started_at

    save_artifacts(model, X.columns.tolist(), metrics, elapsed_seconds)


if __name__ == "__main__":
    main()