import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np


# 1. Load the dataset

DATA_PATH = "data/processed/final_ml_dataset.parquet"

df = pl.read_parquet(DATA_PATH)

print(f"Original shape: {df.shape}")

# Sampling
df = df.sample(n=500_000, seed=42)
print(f"Sampled shape: {df.shape}")


# 2. Split features and target

target_col = "target_win"

X = df.drop(target_col).to_pandas()
y = df[target_col].to_pandas()


# 3. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# 4. Scale features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Train model

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

print("Logistic Regression trained")


# 6. Predict probabilities and classes

y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Lower threshold slightly from 0.50 to 0.48
threshold = 0.48
y_pred = (y_probs >= threshold).astype(int)

print("Sample probabilities:", y_probs[:10])
print(f"Using decision threshold: {threshold}")


# 7. Evaluate

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Logistic Regression Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))