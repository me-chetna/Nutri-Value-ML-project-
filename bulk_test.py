import pandas as pd
import joblib
import numpy as np

df = pd.read_csv("test_data.csv")

model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")

for _, row in df.iterrows():
    sample = np.array([[
        row['calories'], row['sugar'], row['protein'],
        row['fat'], row['sodium'], row['carbohydrates'], row['fiber']
    ]])

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    print(row['ingredients'], "→", "Healthy" if prediction[0] == 1 else "Unhealthy")