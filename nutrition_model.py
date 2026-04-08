# nutrition_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset (replace with your dataset file)
df = pd.read_csv("nutrition.csv")

# Data cleaning: Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Select correct columns
df = df[[
    'Data.Kilocalories',
    'Data.Sugar Total',
    'Data.Protein',
    'Data.Fat.Total Lipid',
    'Data.Major Minerals.Sodium',
    'Data.Carbohydrate',
    'Data.Fiber'
]]

# Rename
df.columns = [
    'calories',
    'sugar',
    'protein',
    'fat',
    'sodium',
    'carbohydrates',
    'fiber'
]

# -----------------------------
# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Display dataset info
print("Dataset Loaded:")
print(df.head())

# Select relevant columns (adjust based on dataset)
features = ['calories', 'sugar', 'protein', 'fat', 'sodium', 'carbohydrates', 'fiber']
df = df[features]

# -----------------------------
# STEP 3: CREATE LABELS
# -----------------------------
def label_food(row):
    if row['sugar'] > 10 or row['fat'] > 15 or row['sodium'] > 400:
        return 0  # Unhealthy
    else:
        return 1  # Healthy

df['label'] = df.apply(label_food, axis=1)


# -----------------------------
# STEP 4: SPLIT DATA
# -----------------------------
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: SCALE DATA
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# STEP 6: TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# STEP 7: EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: SAVE MODEL
# -----------------------------
joblib.dump(model, "nutrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")