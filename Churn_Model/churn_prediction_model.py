"""
Churn Prediction Model
======================
A comprehensive customer churn prediction pipeline using the Churn_Modelling dataset.

Models compared:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost

Outputs:
  - Classification reports and confusion matrices for each model
  - Feature importance plot
  - ROC curves for all models
  - Summary of best model
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ANN - Data and Model", "Churn_Modelling.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
print("\n--- Class Distribution ---")
print(df["Exited"].value_counts(normalize=True).round(3))

print("\n--- Summary Statistics ---")
print(df.describe().round(2))

# Churn rate by Geography and Gender
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df.groupby("Geography")["Exited"].mean().plot.bar(ax=axes[0], color="steelblue")
axes[0].set_title("Churn Rate by Geography")
axes[0].set_ylabel("Churn Rate")
df.groupby("Gender")["Exited"].mean().plot.bar(ax=axes[1], color="coral")
axes[1].set_title("Churn Rate by Gender")
axes[1].set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_churn_by_category.png"), dpi=150)
plt.close()

# Correlation heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
# Drop columns that are not useful for prediction
drop_cols = ["RowNumber", "CustomerId", "Surname"]
X = df.drop(columns=drop_cols + ["Exited"])
y = df["Exited"]

categorical_features = ["Geography", "Gender"]
numerical_features = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)

# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 5. Define Models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    ),
}

# ---------------------------------------------------------------------------
# 6. Train, Evaluate & Compare
# ---------------------------------------------------------------------------
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # Cross-validation on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Fit on full training set and evaluate on test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        "accuracy": acc,
        "roc_auc": auc,
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
        "pipeline": pipe,
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Exited"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix — {name}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.tight_layout()
    fig_cm.savefig(
        os.path.join(OUTPUT_DIR, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=150
    )
    plt.close(fig_cm)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves — Model Comparison")
ax_roc.legend(loc="lower right")
plt.tight_layout()
fig_roc.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
plt.close(fig_roc)

# ---------------------------------------------------------------------------
# 7. Feature Importance (from best tree-based model)
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_pipe = results[best_name]["pipeline"]
best_model = best_pipe.named_steps["classifier"]

# Build feature names after preprocessing
ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
feature_names = numerical_features + cat_feature_names

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    feat_imp.plot.barh(ax=ax_imp, color="teal")
    ax_imp.set_title(f"Feature Importance — {best_name}")
    ax_imp.set_xlabel("Importance")
    plt.tight_layout()
    fig_imp.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig_imp)

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame(
    {
        name: {
            "Test Accuracy": f"{r['accuracy']:.4f}",
            "Test ROC-AUC": f"{r['roc_auc']:.4f}",
            "CV AUC (mean +/- std)": f"{r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f}",
        }
        for name, r in results.items()
    }
).T
print(summary_df.to_string())
print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
print(f"\nPlots saved to: {OUTPUT_DIR}")
