"""Train a three-class Random Forest from nutrition.csv.

Expected CSV columns are the USDA-style columns used in the original project:
Data.Kilocalories, Data.Sugar Total, Data.Protein,
Data.Fat.Total Lipid, Data.Major Minerals.Sodium,
Data.Carbohydrate, Data.Fiber.

Labels are transparent illustrative screening rules, not clinical judgments.
For a research-quality model, replace them with expert-reviewed ground truth.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CSV_PATH = BASE_DIR / "nutrition.csv"
FEATURES = ["calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber"]
SOURCE_COLUMNS = [
    "Data.Kilocalories", "Data.Sugar Total", "Data.Protein",
    "Data.Fat.Total Lipid", "Data.Major Minerals.Sodium",
    "Data.Carbohydrate", "Data.Fiber",
]
LABELS = {0: "Very Poor Choice", 1: "Moderate", 2: "Healthy"}


def assign_screening_label(row):
    """Assign a provisional category using per-100g screening cutoffs."""
    sugar, fat, sodium = row["sugar"], row["fat"], row["sodium"]
    calories, fiber, protein = row["calories"], row["fiber"], row["protein"]
    if sugar >= 22.5 or fat >= 17.5 or sodium >= 600 or calories >= 450:
        return 0
    if sugar <= 5 and fat <= 3 and sodium <= 120 and fiber >= 3 and protein >= 5:
        return 2
    return 1


def train_model():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\nPlace your nutrition.csv beside nutrition_model.py."
        )
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.astype(str).str.strip()
    missing = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("nutrition.csv is missing required columns: " + ", ".join(missing))
    X = df[SOURCE_COLUMNS].copy()
    X.columns = FEATURES
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[(X >= 0).all(axis=1)]
    y = X.apply(assign_screening_label, axis=1)
    counts = y.value_counts().reindex([0, 1, 2], fill_value=0)
    if (counts < 2).any():
        raise ValueError(
            "Training data must produce at least 2 rows in each of the 3 classes. "
            f"Current counts: {dict(zip([LABELS[i] for i in counts.index], counts.values))}. "
            "Add more varied nutrition records or revise the documented screening rules."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Rows used:", len(X))
    print("Class counts:", {LABELS[i]: int(counts[i]) for i in counts.index})
    print("Holdout report:\n", classification_report(y_test, predictions, labels=[0, 1, 2], target_names=[LABELS[i] for i in [0, 1, 2]], zero_division=0))
    print("Confusion matrix (Very Poor, Moderate, Healthy):\n", confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "labels": LABELS}, MODEL_DIR / "nutrition_model.pkl")
    print("Saved:", MODEL_DIR / "nutrition_model.pkl")
    print("NOTE: These categories reproduce the stated screening rules; they are not clinical advice.")


if __name__ == "__main__":
    train_model()
