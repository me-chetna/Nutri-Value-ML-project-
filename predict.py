import joblib
import numpy as np

# Load model + scaler
model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input (change values)
calories = 250
sugar = 12
protein = 5
fat = 10
sodium = 300
carbs = 30
fiber = 3

sample = np.array([[calories, sugar, protein, fat, sodium, carbs, fiber]])

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

# Health Score Function
def health_score(sugar, protein, fat, sodium, fiber):
    score = 100

    score -= sugar * 2
    score -= fat * 1.5
    score -= sodium * 0.05
    score += protein * 2
    score += fiber * 3

    return max(0, min(100, score))


# Use SAME values
score = health_score(sugar, protein, fat, sodium, fiber)
print(f"Health Score: {score}/100")
if score > 70:
    print("🟢 Good choice")
elif score > 40:
    print("🟡 Moderate choice")
else:
    print("🔴 Poor choice")