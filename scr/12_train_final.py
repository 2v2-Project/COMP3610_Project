import os
import json
import joblib
import polars as pl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/final_ml_dataset.parquet"
TARGET_COL = "target_win"
DROP_COLS = [TARGET_COL, "match_id"]

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42
THRESHOLD = 0.50

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "columns.json")
METRICS_PATH = os.path.join(MODEL_DIR, "random_forest_metrics.json")


def main():
    print("=== FINAL RANDOM FOREST TRAINING ===")

    # 1. Load dataset
    df = pl.read_parquet(DATA_PATH)
    print(f"Original shape: {df.shape}")

    # 2. Sample largest stable subset
    df = df.sample(n=SAMPLE_SIZE, seed=RANDOM_STATE)
    print(f"Sampled shape: {df.shape}")

    # 3. Split features and target
    X = df.drop(DROP_COLS).to_pandas()
    y = df[TARGET_COL].to_pandas()

    print(f"Feature matrix shape: {X.shape}")

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. Train selected RF model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,  # safer than -1 for memory
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    print("Random Forest trained")

    # 6. Predict
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= THRESHOLD).astype(int)

    print("Sample probabilities:", y_probs[:10])
    print(f"Using decision threshold: {THRESHOLD}")

    # 7. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Random Forest Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values(ascending=False).head(10))

    # 9. Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(SCHEMA_PATH, "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    metrics = {
        "model_name": "final_random_forest",
        "sample_size": SAMPLE_SIZE,
        "threshold": THRESHOLD,
        "accuracy": round(float(accuracy), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "parameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        }
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Saved Artifacts ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Feature schema: {SCHEMA_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()