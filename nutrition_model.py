# nutrition_model.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
df = pd.read_csv("nutrition.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Select required columns
df = df[[
    'Data.Kilocalories',
    'Data.Sugar Total',
    'Data.Protein',
    'Data.Fat.Total Lipid',
    'Data.Major Minerals.Sodium',
    'Data.Carbohydrate',
    'Data.Fiber'
]]

# Rename columns
df.columns = [
    'calories',
    'sugar',
    'protein',
    'fat',
    'sodium',
    'carbohydrates',
    'fiber'
]

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print("Dataset Loaded:")
print(df.head())

# -----------------------------
# STEP 2: CREATE LABELS
# -----------------------------
def label_food(row):
    if row['sugar'] > 10 or row['fat'] > 15 or row['sodium'] > 400:
        label = 0
    else:
        label = 1
    
    # add slight randomness
    if random.random() < 0.1:
        label = 1 - label
    
    return label

df['label'] = df.apply(label_food, axis=1)

# -----------------------------
# STEP 3: SPLIT DATA
# -----------------------------
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# STEP 4: SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# STEP 5: TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# STEP 6: EVALUATE MODEL
# -----------------------------

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])  # baseline

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.show()

# -----------------------------
# STEP 7: SAVE MODEL
# -----------------------------
joblib.dump(model, "nutrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")