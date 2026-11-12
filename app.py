"""Streamlit app: nutrition-label OCR/manual verification + Random Forest."""
from pathlib import Path
import hashlib

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from database import initialize_database, seed_products, get_history, save_scan, get_alternatives, get_connection
from ocr_utils import extract_text, extract_nutrition

st.set_page_config(page_title="Smart Food Analyzer", page_icon="🥗", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "nutrition_model.pkl"
CNN_PATH = BASE_DIR / "models" / "cnn_nutrition_model.pth"
CNN_SCALER_PATH = BASE_DIR / "models" / "cnn_target_scaler.pkl"
FEATURES = ["calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber"]
LABELS = {0: "Very Poor Choice", 1: "Moderate", 2: "Healthy"}
UNITS = {"calories": "kcal", "sugar": "g", "protein": "g", "fat": "g", "sodium": "mg", "carbohydrates": "g", "fiber": "g"}

initialize_database()
seed_products()

@st.cache_resource
def load_model(path_string):
    bundle = joblib.load(path_string)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("Invalid model file. Run nutrition_model.py to retrain it.")
    if bundle["features"] != FEATURES:
        raise ValueError("Model feature order does not match the application.")
    return bundle

@st.cache_resource
def load_cnn(path_string):
    import torch
    from cnn_model import FoodNutritionCNN
    cnn = FoodNutritionCNN(output_features=len(FEATURES))
    state = torch.load(path_string, map_location="cpu")
    cnn.load_state_dict(state)
    cnn.eval()
    target_scaler = joblib.load(str(CNN_SCALER_PATH))
    return cnn, target_scaler

def predict_cnn_values(image_bytes):
    import torch
    from io import BytesIO
    from PIL import Image, ImageOps
    from torchvision import transforms
    cnn, target_scaler = load_cnn(str(CNN_PATH))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(BytesIO(image_bytes)) as image:
        tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        scaled_values = cnn(tensor).cpu().numpy()
    values = target_scaler.inverse_transform(scaled_values)[0]
    return {feature: max(0.0, float(value)) for feature, value in zip(FEATURES, values)}

if not MODEL_PATH.is_file():
    st.error("Random Forest model not found. First place nutrition.csv in this folder and run: python nutrition_model.py")
    st.stop()
try:
    model_bundle = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()
model = model_bundle["model"]

st.title("🥗 Smart Food Health Analyzer")
st.caption("Scan a packaged-food nutrition label, verify the extracted values, and classify it with a three-class Random Forest.")
page = st.sidebar.radio("Navigation", ["Home", "Scan Food", "History"])

if page == "Home":
    st.header("Welcome to Smart Food Analyzer")
    history = get_history()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scans", len(history))
    c2.metric("Healthy", int((history["prediction"] == "Healthy").sum()) if not history.empty else 0)
    c3.metric("Very poor", int((history["prediction"] == "Very Poor Choice").sum()) if not history.empty else 0)
    st.subheader("Recommended demo foods")
    with get_connection() as conn:
        products = pd.read_sql_query("SELECT * FROM products WHERE category='Healthy'", conn)
    if products.empty:
        st.info("No demo products are available yet.")
    else:
        st.dataframe(products.drop(columns=["id", "category"], errors="ignore"), use_container_width=True, hide_index=True)
    if not history.empty:
        st.subheader("Scanning activity")
        st.plotly_chart(px.bar(history, x="product_name", color="prediction"), use_container_width=True)

elif page == "Scan Food":
    st.header("📷 Scan nutrition label")
    source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
    uploaded = st.file_uploader("Upload the back-side nutrition label", type=["jpg", "jpeg", "png"]) if source == "Upload a photo" else st.camera_input("Take a photo of the nutrition label")

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image_id = hashlib.sha256(image_bytes).hexdigest()
        st.image(image_bytes, caption="Uploaded nutrition label", width=480)
        if st.session_state.get("image_id") != image_id:
            st.session_state.image_id = image_id
            st.session_state.extracted = None
            st.session_state.ocr_text = ""
        if st.button("Read nutrition label with OCR", type="primary"):
            try:
                text = extract_text(image_bytes)
                st.session_state.ocr_text = text
                st.session_state.extracted = extract_nutrition(text)
                if not text.strip():
                    st.warning("No text was detected. Enter the values manually.")
                elif all(v is None for v in st.session_state.extracted.values()):
                    st.warning("Text was detected, but nutrition values were not recognized. Enter them manually.")
                else:
                    st.success("OCR completed. Please verify every value and unit below.")
            except Exception as exc:
                st.error(f"OCR failed: {exc}. You can still enter the values manually.")
                st.session_state.extracted = {feature: None for feature in FEATURES}

        if CNN_PATH.is_file() and CNN_SCALER_PATH.is_file():
            if st.button("Generate CNN nutrition estimates"):
                try:
                    st.session_state.extracted = predict_cnn_values(image_bytes)
                    st.success("CNN estimates generated. Verify every value against the printed label before classification.")
                except Exception as exc:
                    st.error(f"CNN prediction failed: {exc}")
        else:
            st.caption("CNN estimates are currently disabled because trained CNN weights are not present. Train it with labelled nutrition-label images using train_cnn.py.")

        if st.session_state.get("extracted") is not None:
            if st.session_state.ocr_text:
                with st.expander("View recognized text"):
                    st.text(st.session_state.ocr_text)
            st.subheader("Verify / correct nutrition values")
            st.info("Enter values exactly as printed on the package. The app converts per-serving values to per 100 g/ml before prediction. OCR results must be checked by you.")
            image_key = st.session_state.image_id
            basis = st.radio("Values shown on package", ["Per 100 g/ml", "Per serving"], horizontal=True, key=f"basis_{image_key}")
            serving_size = None
            if basis == "Per serving":
                serving_size = st.number_input("Serving size (g or ml)", min_value=0.1, value=30.0, step=0.1, key=f"serving_{image_key}")
            with st.form(f"nutrition_form_{image_key}"):
                product_name = st.text_input("Product name")
                price = st.number_input("Product price (₹)", min_value=0.0, value=50.0, step=1.0)
                entered = {}
                columns = st.columns(2)
                for index, feature in enumerate(FEATURES):
                    default = st.session_state.extracted.get(feature)
                    with columns[index % 2]:
                        entered[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()} ({UNITS[feature]})",
                            min_value=0.0,
                            value=float(default) if default is not None else 0.0,
                            step=0.1,
                            format="%.2f",
                            help="If OCR did not find this value, replace 0 with the value printed on the label.",
                        )
                ingredients = st.text_area("Ingredients (optional)")
                submitted = st.form_submit_button("Analyze with Random Forest", type="primary")

            if submitted:
                if basis == "Per serving" and (serving_size is None or serving_size <= 0):
                    st.error("Serving size must be greater than zero.")
                else:
                    factor = 1.0 if basis == "Per 100 g/ml" else 100.0 / serving_size
                    per_100 = {feature: float(entered[feature]) * factor for feature in FEATURES}
                    # Basic sanity checks catch common OCR/unit mistakes before model inference.
                    if per_100["sodium"] > 10000 or per_100["calories"] > 1000 or any(per_100[k] > 100 for k in ["sugar", "protein", "fat", "carbohydrates", "fiber"]):
                        st.error("At least one value looks unusually high per 100 g/ml. Check the numbers and units before analyzing.")
                    else:
                        sample = pd.DataFrame([[per_100[f] for f in FEATURES]], columns=FEATURES)
                        predicted_class = int(model.predict(sample)[0])
                        prediction = LABELS.get(predicted_class, str(predicted_class))
                        probabilities = model.predict_proba(sample)[0]
                        confidence = float(max(probabilities) * 100)
                        st.subheader("Analysis result (per 100 g/ml)")
                        a, b = st.columns(2)
                        a.metric("Random Forest category", prediction)
                        b.metric("Model confidence", f"{confidence:.1f}%")
                        st.caption("Confidence is the model's class probability, not a guarantee of nutritional quality. Categories are based on illustrative screening labels used during training.")
                        st.dataframe(pd.DataFrame({"Nutrient": FEATURES, "Value per 100 g/ml": [round(per_100[f], 2) for f in FEATURES], "Unit": [UNITS[f] for f in FEATURES]}), hide_index=True, use_container_width=True)
                        if prediction == "Healthy":
                            st.success("This product falls into the model's Healthy category.")
                        elif prediction == "Moderate":
                            st.warning("This product falls into the model's Moderate category. Compare it with alternatives and serving sizes.")
                        else:
                            st.error("This product falls into the model's Very Poor Choice category under the project's screening rules.")
                        save_scan({"name": product_name.strip() or "Unnamed Product", "price": price, **per_100, "prediction": prediction, "confidence": confidence, "ingredients": ingredients})
                        st.success("Scan saved to history.")
                        if prediction != "Healthy":
                            alternatives = get_alternatives(price)
                            st.subheader("Lower-price demo alternatives")
                            if alternatives.empty:
                                st.info("No seeded alternatives were found at or below this price.")
                            else:
                                st.dataframe(alternatives, hide_index=True, use_container_width=True)
    else:
        st.info("Upload or capture a clear photo of the nutrition information panel to begin.")

else:
    st.header("📊 Scan history")
    history = get_history()
    if history.empty:
        st.info("You have not scanned any products yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.plotly_chart(px.pie(history, names="prediction", title="Food category distribution"), use_container_width=True)
