"""
Task 34A — Train XGBoost Model
===============================
Uses the same feature set, sample size, and train/test split as
12_train_final.py (Random Forest) for direct comparison.

Includes light hyper-parameter tuning via RandomizedSearchCV (cheap
but effective) and saves all artifacts to models/.
"""

import os
import json
import time
import joblib

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import xgboost as xgb

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/final_ml_dataset.parquet"
TARGET_COL = "target_win"
DROP_COLS = [TARGET_COL, "match_id"]

SAMPLE_SIZE = 1_000_000   # same as 12_train_final.py
RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.50

# Tuning budget — small enough to finish in minutes, large enough to
# find a strong configuration.
TUNE_N_ITER = 12
TUNE_CV_FOLDS = 3

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "columns.json")
METRICS_PATH = os.path.join(MODEL_DIR, "xgboost_metrics.json")


# =========================
# HELPERS
# =========================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    print("=== Loading dataset ===")
    df = pl.read_parquet(DATA_PATH)
    print(f"Original shape: {df.shape}")

    df = df.sample(n=SAMPLE_SIZE, seed=RANDOM_STATE)
    print(f"Sampled shape:  {df.shape}")

    pdf = df.to_pandas()

    # Downcast to reduce memory
    for col in pdf.select_dtypes(include=["int64"]).columns:
        pdf[col] = pd.to_numeric(pdf[col], downcast="integer")
    for col in pdf.select_dtypes(include=["float64"]).columns:
        pdf[col] = pd.to_numeric(pdf[col], downcast="float")

    mem_mb = pdf.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory usage:   {mem_mb:.1f} MB")

    X = pdf.drop(columns=DROP_COLS)
    y = pdf[TARGET_COL]
    print(f"Feature matrix: {X.shape}")
    return X, y


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """Light randomised search over XGBoost hyper-parameters."""
    print("\n=== Hyper-parameter tuning (RandomizedSearchCV) ===")

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",          # fast histogram-based splits
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    param_dist = {
        "n_estimators":     [200, 300, 500],
        "max_depth":        [4, 6, 8, 10],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.3],
        "reg_alpha":        [0, 0.1, 1.0],
        "reg_lambda":       [1, 2, 5],
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=TUNE_N_ITER,
        cv=TUNE_CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,            # outer parallelism = 1 (XGB already parallel)
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params:     {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    print("\n=== XGBoost Evaluation ===")
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Threshold: {threshold}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def save_artifacts(model, columns: list[str], best_params: dict,
                   best_cv_auc: float, test_metrics: dict, elapsed: float):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(SCHEMA_PATH, "w") as f:
        json.dump(columns, f, indent=2)

    meta = {
        "model_name": "xgboost",
        "sample_size": SAMPLE_SIZE,
        "threshold": THRESHOLD,
        "tune_n_iter": TUNE_N_ITER,
        "tune_cv_folds": TUNE_CV_FOLDS,
        "best_cv_roc_auc": round(float(best_cv_auc), 4),
        **test_metrics,
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in best_params.items()},
        "training_time_seconds": round(elapsed, 1),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Saved Artifacts ===")
    print(f"Model:          {MODEL_PATH}")
    print(f"Feature schema: {SCHEMA_PATH}")
    print(f"Metrics:        {METRICS_PATH}")


def print_feature_importance(model, columns: list[str], top_n: int = 15):
    importances = pd.Series(model.feature_importances_, index=columns)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop {top_n} Most Important Features:")
    print(importances.head(top_n).to_string())


# =========================
# MAIN
# =========================

def main():
    print("=" * 60)
    print("  Task 34A — XGBoost Training Pipeline")
    print("=" * 60)

    t0 = time.time()

    # 1. Load data
    X, y = load_data()

    # 2. Same train/test split as all other models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

    # 3. Tune + train
    best_model, best_params, best_cv_auc = tune_xgboost(X_train, y_train)

    # 4. Evaluate on hold-out
    test_metrics = evaluate(best_model, X_test, y_test, THRESHOLD)

    # 5. Feature importance
    print_feature_importance(best_model, X.columns.tolist())

    # 6. Save
    elapsed = time.time() - t0
    save_artifacts(best_model, X.columns.tolist(), best_params,
                   best_cv_auc, test_metrics, elapsed)

    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
