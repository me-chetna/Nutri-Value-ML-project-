import streamlit as st
import joblib
import numpy as np

model = joblib.load("nutrition_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🍎 Nutrition Health Checker")

calories = st.number_input("Calories")
sugar = st.number_input("Sugar")
protein = st.number_input("Protein")
fat = st.number_input("Fat")
sodium = st.number_input("Sodium")
carbs = st.number_input("Carbohydrates")
fiber = st.number_input("Fiber")

if st.button("Check Health"):
    sample = np.array([[calories, sugar, protein, fat, sodium, carbs, fiber]])
    sample_scaled = scaler.transform(sample)

    prediction_proba = model.predict_proba(sample_scaled)
    score = prediction_proba[0][1] * 100

    if prediction_proba[0][1] > 0.5:
        st.success("Healthy ✅")
    else:
        st.error("Unhealthy ❌")
    if score > 70:
        st.write("🟢 Good choice")
    elif score > 40:
        st.write("🟡 Moderate choice")
    else:
        st.write("🔴 Poor choice")