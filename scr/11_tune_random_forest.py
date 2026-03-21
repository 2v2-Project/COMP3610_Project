import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/final_ml_dataset.parquet"   # change if needed
TARGET_COL = "target_win"                               # change if needed
DROP_COLS = [TARGET_COL, "match_id"]                    # remove any non-feature cols here
MODEL_DIR = "models"

SAMPLE_SIZE = 500_000      # keep this modest to reduce load
TEST_SIZE = 0.2
RANDOM_STATE = 42

# very small search space to keep tuning cheap
N_ITER = 4
CV_FOLDS = 2


def load_data():
    print("=== Loading dataset ===")
    df = pd.read_parquet(DATA_PATH)
    print("Original shape:", df.shape)

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print("Sampled shape:", df.shape)

    # Downcast numeric columns to reduce memory
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    print("Memory usage (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]

    print("Feature matrix shape:", X.shape)
    return X, y


def tune_random_forest(X_train, y_train):
    print("\n=== Tuning Random Forest ===")

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [10, 20, None]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    return search.best_estimator_, search.best_params_


def evaluate_model(model, X_test, y_test):
    print("\n=== Evaluating Tuned Random Forest ===")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")


def save_artifacts(model, columns, best_params):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "random_forest_tuned.pkl")
    cols_path = os.path.join(MODEL_DIR, "columns.json")
    params_path = os.path.join(MODEL_DIR, "random_forest_tuned_params.json")

    joblib.dump(model, model_path)

    with open(cols_path, "w") as f:
        json.dump(columns, f, indent=2)

    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print("\n=== Saved artifacts ===")
    print("Model:", model_path)
    print("Columns:", cols_path)
    print("Params:", params_path)


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Train:", X_train.shape, "Test:", X_test.shape)

    best_model, best_params = tune_random_forest(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    # save_artifacts(best_model, X.columns.tolist(), best_params)


if __name__ == "__main__":
    main()

# Model comparison notes (latest observed runs)
# Logistic Regression (scr/09_train_logistic_regression.py, sample=500k):
# - Accuracy: 0.5471
# - F1 Score: 0.5901
# - ROC-AUC: 0.5741
#
# Random Forest baseline (scr/10_train_random_forest.py, sample=500k):
# - Accuracy: 0.5645
# - F1 Score: 0.5633
# - ROC-AUC: 0.5912
#
# Tuned Random Forest (this script, sample=200k, cv=2, n_iter=6):
# - Best params: n_estimators=100, max_depth=20
# - Accuracy: 0.5679
# - F1 Score: 0.5136
# - ROC-AUC: 0.5956
#
# Findings:
# - Best ROC-AUC comes from tuned RF, then baseline RF, then logistic regression.
# - Logistic regression has the highest F1 in these runs.
# - Tuned RF improves ranking/discrimination (ROC-AUC) but predicts fewer positives,
#   which lowers recall/F1 at threshold=0.50.
# - Best Model: Baseline RF