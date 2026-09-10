"""
train.py — Bank Marketing Term Deposit Prediction
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
# Find the CSV automatically, no matter where this script is run from,
# and no matter whether the CSV sits flat next to it or inside a
# "bank-additional" subfolder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.join(SCRIPT_DIR, "bank-additional-full.csv"),
    os.path.join(SCRIPT_DIR, "bank-additional", "bank-additional-full.csv"),
    "bank-additional-full.csv",
    "bank-additional/bank-additional-full.csv",
]

csv_path = next((p for p in CANDIDATE_PATHS if os.path.isfile(p)), None)
if csv_path is None:
    raise FileNotFoundError(
        "Couldn't find bank-additional-full.csv. Make sure it's in the same "
        "folder as this script (or in a 'bank-additional' subfolder)."
    )

print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path, sep=";")

df = df.drop(columns=["duration"])  # data leakage — see note above

y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

# Label-encode categoricals (simple approach; one-hot would also work)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("=== Random Forest ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, rf_proba):.3f}")

# ---------------------------------------------------------------
# 3. XGBoost
# ---------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

print("\n=== XGBoost ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, xgb_proba):.3f}")

# ---------------------------------------------------------------
# 4. Feature selection: RFE
# ---------------------------------------------------------------
rfe_estimator = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=8)
rfe.fit(X_train_scaled, y_train)
selected_features = X.columns[rfe.support_].tolist()
print("\n=== RFE: top 8 selected features ===")
print(selected_features)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Random Forest feature importances (top 10) ===")
print(importances.head(10).round(3).to_string())

# ---------------------------------------------------------------
# 5. PCA
# ---------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
print(f"\n=== PCA ===")
print(f"Explained variance (2 components): {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap="coolwarm", alpha=0.3, s=8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
plt.title("PCA of bank marketing customers (colored by subscription outcome)")
plt.colorbar(scatter, label="subscribed")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "pca_plot.png"), dpi=150)
print("Saved pca_plot.png")

plt.figure(figsize=(7, 5))
importances.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Random Forest feature importances")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "feature_importance.png"), dpi=150)
print("Saved feature_importance.png")

# ---------------------------------------------------------------
# 6. Confusion matrix + full report (best model)
# ---------------------------------------------------------------
best_auc_rf = roc_auc_score(y_test, rf_proba)
best_auc_xgb = roc_auc_score(y_test, xgb_proba)
best_pred = xgb_pred if best_auc_xgb > best_auc_rf else rf_pred
best_name = "XGBoost" if best_auc_xgb > best_auc_rf else "Random Forest"

print(f"\n=== Classification report ({best_name}) ===")
print(classification_report(y_test, best_pred, target_names=["no", "yes"]))

cm = confusion_matrix(y_test, best_pred)
print("Confusion matrix:")
print(cm)
