"""
Heart Disease Risk Classifier - training script
Run:  python train_model.py
Outputs: model.joblib, correlation_heatmap.png, confusion_matrix.png
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65

# 1. Load data ---------------------------------------------------------
df = pd.read_csv("heart.csv")
df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce")
print("Dataset shape:", df.shape)
print("Class distribution:\n", df["target"].value_counts())

# 2. EDA: correlation heatmap -----------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Feature Correlation Heatmap - Heart Disease Dataset")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# 3. Split FIRST, then preprocess (avoids data leakage) ---------------
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

categorical_cols = ["cp", "restecg", "slope", "thal"]
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
passthrough_cols = ["sex", "fbs", "exang", "ca"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("pass", SimpleImputer(strategy="median"), passthrough_cols),
])

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
])
pipe.fit(X_train, y_train)

# 4. Three-tier risk labelling ----------------------------------------
def risk_label(p):
    if p < LOW_THRESHOLD:
        return "Low Risk"
    if p > HIGH_THRESHOLD:
        return "High Risk"
    return "Moderate Risk"

probas = pipe.predict_proba(X_test)[:, 1]
y_pred_risk = [risk_label(p) for p in probas]
y_test_risk = ["High Risk" if v == 1 else "Low Risk" for v in y_test]

# 5. Evaluation --------------------------------------------------------
print(f"\nBinary accuracy: {accuracy_score(y_test, pipe.predict(X_test)):.4f}")
cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
print(f"5-fold cross-validated accuracy: {cv.mean():.4f}  (more reliable than one split)")

print("\n--- Three-tier report ---")
print(classification_report(y_test_risk, y_pred_risk, zero_division=0))

labels = ["Low Risk", "Moderate Risk", "High Risk"]
cm = confusion_matrix(y_test_risk, y_pred_risk, labels=labels)
print("Confusion matrix (rows=true, cols=predicted):\n", cm)

plt.figure(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Heart Disease Risk Classifier")
plt.xlabel("Predicted Risk Level")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 6. Save the whole pipeline -------------------------------------------
joblib.dump({"pipeline": pipe, "low": LOW_THRESHOLD, "high": HIGH_THRESHOLD,
             "columns": list(X.columns)}, "model.joblib")
print("\nSaved model.joblib")
