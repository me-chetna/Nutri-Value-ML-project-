import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("test_data.csv")

# Load model
model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")

predictions = []

# Generate predictions
for _, row in df.iterrows():
    sample = np.array([[
        row['calories'], row['sugar'], row['protein'],
        row['fat'], row['sodium'], row['carbohydrates'], row['fiber']
    ]])

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]

    predictions.append(pred)

df['prediction'] = predictions

# -----------------------------
# GRAPH 1: Healthy vs Unhealthy Distribution
# -----------------------------
counts = df['prediction'].value_counts()

plt.figure()
plt.pie(counts, labels=["Unhealthy", "Healthy"], autopct='%1.1f%%')
plt.title("Food Health Distribution")
plt.show()

# -----------------------------
# GRAPH 2: Healthy vs Unhealthy Count
# -----------------------------
plt.figure()
counts.plot(kind='bar')
plt.title("Healthy vs Unhealthy Foods")
plt.xlabel("Class (1=Healthy, 0=Unhealthy)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# GRAPH 3: Sugar vs Health
# -----------------------------
plt.figure()
for label in [0, 1]:
    subset = df[df['prediction'] == label]
    plt.scatter(subset['sugar'], subset['fat'], label=f"{'Healthy' if label==1 else 'Unhealthy'}")

plt.xlabel("Sugar")
plt.ylabel("Fat")
plt.title("Sugar vs Fat (Health Trend)")
plt.legend()
plt.show()

