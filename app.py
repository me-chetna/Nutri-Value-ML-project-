import streamlit as st
import joblib
import numpy as np

# Load model + scaler
model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Ingredient Risk Dictionary
# ---------------------------
harmful_ingredients = {
    "high fructose corn syrup": "Very high sugar → increases obesity risk",
    "aspartame": "Artificial sweetener → controversial health effects",
    "monosodium glutamate": "May cause headaches in sensitive people",
    "sodium benzoate": "Preservative → may form harmful compounds",
    "trans fat": "Increases bad cholesterol (very harmful)",
    "artificial color": "Linked to hyperactivity in some cases",
    "palm oil": "High saturated fat → not heart friendly",
    "refined sugar": "Empty calories → spikes blood sugar"
}

# ---------------------------
# Functions
# ---------------------------
def analyze_ingredients(ingredient_text):
    ingredient_text = ingredient_text.lower()
    warnings = []

    for ingredient, reason in harmful_ingredients.items():
        if ingredient in ingredient_text:
            warnings.append(f"{ingredient} → {reason}")

    return warnings


def health_score(sugar, protein, fat, sodium, fiber):
    score = 100
    score -= sugar * 2
    score -= fat * 1.5
    score -= sodium * 0.05
    score += protein * 2
    score += fiber * 3
    return max(0, min(100, score))


# ---------------------------
# UI
# ---------------------------
st.title("🍎 Smart Food Health Analyzer")

calories = st.number_input("Calories")
sugar = st.number_input("Sugar")
protein = st.number_input("Protein")
fat = st.number_input("Fat")
sodium = st.number_input("Sodium")
carbs = st.number_input("Carbohydrates")
fiber = st.number_input("Fiber")

st.subheader("Ingredients")
ingredients = st.text_area("Enter ingredients (comma separated)")

# ---------------------------
# Button Action
# ---------------------------
if st.button("Analyze Food"):

    # ML Prediction
    sample = np.array([[calories, sugar, protein, fat, sodium, carbs, fiber]])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    # Health Score
    score = health_score(sugar, protein, fat, sodium, fiber)

    # Ingredient Analysis
    warnings = analyze_ingredients(ingredients)

    st.subheader("Results")

    # Prediction
    if prediction[0] == 1:
        st.success("Healthy ✅")
    else:
        st.error("Unhealthy ❌")

    # Score
    st.write(f"### Health Score: {score}/100")

    if score > 80:
        st.write("🟢 Good choice")
    elif score > 50:
        st.write("🟡 Moderate choice")
    else:
        st.write("🔴 Poor choice")

    # Ingredient Analysis
    st.subheader("Ingredient Analysis")

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No harmful ingredients detected")