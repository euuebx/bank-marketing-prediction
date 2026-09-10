"""
train.py - Bank Marketing Term Deposit Prediction
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


# Load the data

script_dir = os.path.dirname(os.path.abspath(__file__))

files = [
    os.path.join(script_dir, "bank-additional-full.csv"),
    os.path.join(script_dir, "bank-additional", "bank-additional-full.csv"),
    "bank-additional-full.csv",
    "bank-additional/bank-additional-full.csv"
]

csv_path = next((file for file in files if os.path.isfile(file)), None)

if csv_path is None:
    raise FileNotFoundError(
        "Could not find bank-additional-full.csv. "
        "Make sure the file is in the same folder as this script."
    )

print("Loading data from:", csv_path)

df = pd.read_csv(csv_path, sep=";")


# Remove duration because it would cause data leakage

df = df.drop(columns=["duration"])

y = (df["y"] == "yes").astype(int)
X = df.drop(columns=["y"])


# Encode categorical columns

categorical_columns = X.select_dtypes(include="object").columns

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    encoders[column] = encoder


# Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Random Forest

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print("\n=== Random Forest ===")
print(f"Accuracy: {rf_accuracy:.3f}")
print(f"ROC-AUC:  {rf_auc:.3f}")


# XGBoost

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train_scaled, y_train)

xgb_pred = xgb.predict(X_test_scaled)
xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

print("\n=== XGBoost ===")
print(f"Accuracy: {xgb_accuracy:.3f}")
print(f"ROC-AUC:  {xgb_auc:.3f}")


# RFE feature selection

rfe_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

rfe = RFE(
    estimator=rfe_model,
    n_features_to_select=8
)

rfe.fit(X_train_scaled, y_train)

selected_features = X.columns[rfe.support_].tolist()

print("\n=== RFE: top 8 selected features ===")
print(selected_features)


# Feature importance

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n=== Random Forest feature importances ===")
print(importances.head(10).round(3).to_string())


# PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

explained_variance = pca.explained_variance_ratio_.sum()

print("\n=== PCA ===")
print(
    f"Explained variance (2 components): "
    f"{explained_variance:.2%}"
)


# PCA plot

plt.figure(figsize=(7, 5))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.3,
    s=8
)

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
)

plt.title("PCA of bank marketing customers")

plt.colorbar(scatter, label="subscribed")

plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, "pca_plot.png"),
    dpi=150
)

plt.close()

print("Saved pca_plot.png")


# Feature importance plot

plt.figure(figsize=(7, 5))

importances.head(10).plot(kind="barh")

plt.gca().invert_yaxis()

plt.title("Top 10 Random Forest feature importances")

plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, "feature_importance.png"),
    dpi=150
)

plt.close()

print("Saved feature_importance.png")


# Choose the better model

if xgb_auc > rf_auc:
    best_name = "XGBoost"
    best_pred = xgb_pred
else:
    best_name = "Random Forest"
    best_pred = rf_pred


# Final results

print(f"\n=== Classification report ({best_name}) ===")

print(
    classification_report(
        y_test,
        best_pred,
        target_names=["no", "yes"]
    )
)

cm = confusion_matrix(y_test, best_pred)

print("Confusion matrix:")
print(cm)
