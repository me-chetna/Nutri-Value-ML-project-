"""Command-line example for the trained Random Forest model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "models" / "nutrition_model.pkl")
features = bundle["features"]
labels = bundle["labels"]
# Replace this example with verified values per 100 g/ml.
example = {"calories": 250, "sugar": 12, "protein": 5, "fat": 10, "sodium": 300, "carbohydrates": 30, "fiber": 3}
sample = pd.DataFrame([[example[name] for name in features]], columns=features)
prediction = int(bundle["model"].predict(sample)[0])
probabilities = bundle["model"].predict_proba(sample)[0]
print("Category:", labels[prediction])
print("Model class probabilities:", {labels[int(c)]: round(float(p), 3) for c, p in zip(bundle["model"].classes_, probabilities)})
