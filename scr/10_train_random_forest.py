import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd


# 1. Load the dataset
DATA_PATH = "data/processed/final_ml_dataset.parquet"

df = pl.read_parquet(DATA_PATH)

print(f"Original shape: {df.shape}")

# Same sampling strategy as logistic regression for comparability
df = df.sample(n=500_000, seed=42)
print(f"Sampled shape: {df.shape}")


# 2. Split features and target

target_col = "target_win"
drop_cols = [target_col, "match_id"]

X = df.drop(drop_cols).to_pandas()
y = df[target_col].to_pandas()

print(f"Feature matrix shape: {X.shape}")


# 3. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# 4. Train model 

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("Random Forest trained")


# 5. Predict probabilities and classes

y_probs = model.predict_proba(X_test)[:, 1]

threshold = 0.50
y_pred = (y_probs >= threshold).astype(int)

print("Sample probabilities:", y_probs[:10])
print(f"Using decision threshold: {threshold}")


# 6. Evaluate model

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


# 7. Feature Importance

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values(ascending=False).head(10))

# Model comparison to add into documentation later

# Logistic Regression:
# - Accuracy: 0.5511
# - F1 Score: 0.5917
# - ROC-AUC: 0.5764

# Random Forest:
# - Accuracy: 0.5655
# - F1 Score: 0.5631
# - ROC-AUC: 0.5918
