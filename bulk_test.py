import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(BASE_DIR / "test_data.csv")
model = joblib.load(MODEL_DIR / "nutrition_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

for _, row in df.iterrows():
    sample = pd.DataFrame([[
        row["calories"], row["sugar"], row["protein"],
        row["fat"], row["sodium"], row["carbohydrates"], row["fiber"]
    ]], columns=[
        "calories", "sugar", "protein", "fat",
        "sodium", "carbohydrates", "fiber"
    ])

    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    print(row["ingredients"], "->", "Healthy" if prediction[0] == 1 else "Unhealthy")