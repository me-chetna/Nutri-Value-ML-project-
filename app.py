
"""Smart Food Analyzer - Streamlit Application"""

from pathlib import Path
import hashlib
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import (
    initialize_database,
    seed_products,
    get_history,
    save_scan,
    get_alternatives,
    get_connection,
)

from ocr_utils import extract_text, extract_nutrition
from fuzzy_health import calculate_health_score


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Food Analyzer",
    page_icon="🥗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

CNN_PATH = BASE_DIR / "models" / "cnn_nutrition_model.pth"
CNN_SCALER_PATH = BASE_DIR / "models" / "cnn_target_scaler.pkl"

FEATURES = [
    "calories",
    "sugar",
    "protein",
    "fat",
    "sodium",
    "carbohydrates",
    "fiber",
]

UNITS = {
    "calories": "kcal",
    "sugar": "g",
    "protein": "g",
    "fat": "g",
    "sodium": "mg",
    "carbohydrates": "g",
    "fiber": "g",
}

initialize_database()
seed_products()


# --------------------------------------------------
# CNN MODEL FUNCTIONS
# --------------------------------------------------

@st.cache_resource
def load_cnn(path_string):
    import torch
    import joblib

    from cnn_model import FoodNutritionCNN

    cnn = FoodNutritionCNN(output_features=len(FEATURES))

    state = torch.load(path_string, map_location="cpu")
    cnn.load_state_dict(state)
    cnn.eval()

    target_scaler = joblib.load(str(CNN_SCALER_PATH))

    return cnn, target_scaler


def predict_cnn_values(image_bytes):
    import torch

    from PIL import Image, ImageOps
    from torchvision import transforms

    cnn, target_scaler = load_cnn(str(CNN_PATH))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        scaled_values = cnn(tensor).cpu().numpy()

    values = target_scaler.inverse_transform(scaled_values)[0]

    return {
        feature: max(0.0, float(value))
        for feature, value in zip(FEATURES, values)
    }


# --------------------------------------------------
# CIRCULAR HEALTH SCORE
# --------------------------------------------------

def show_health_meter(score, category):

    if score >= 80:
        color = "#16A34A"

    elif score >= 65:
        color = "#65A30D"

    elif score >= 45:
        color = "#EAB308"

    elif score >= 25:
        color = "#F97316"

    else:
        color = "#DC2626"

    fig = go.Figure(
        go.Pie(
            values=[score, 100 - score],
            hole=0.78,
            marker=dict(
                colors=[color, "#E5E7EB"],
                line=dict(color="white", width=2),
            ),
            sort=False,
            direction="clockwise",
            textinfo="none",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        showlegend=False,
        height=270,
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[
            dict(
                text=(
                    f"<b>{score:.2f}%</b>"
                    f"<br><span style='font-size:14px'>{category}</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=25, color=color),
            )
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# --------------------------------------------------
# HEALTH CATEGORY MESSAGE
# --------------------------------------------------

def show_category_message(category):

    if category == "Very Healthy":
        st.success(
            "Excellent! This product has a very healthy "
            "nutrition profile according to the project score."
        )

    elif category == "Healthy":
        st.success(
            "This product has a healthy nutrition profile."
        )

    elif category == "Moderate":
        st.warning(
            "This product has a moderate nutrition profile. "
            "Compare it with other products."
        )

    elif category == "Poor":
        st.warning(
            "Consider comparing this product with options "
            "containing less sugar or sodium."
        )

    else:
        st.error(
            "This product receives a very poor score under "
            "the project's nutrition screening rules."
        )


# --------------------------------------------------
# APP TITLE AND NAVIGATION
# --------------------------------------------------

st.title("🥗 Smart Food Health Analyzer")

st.caption(
    "Scan packaged-food nutrition labels, check nutrition values, "
    "and calculate a health score using fuzzy logic."
)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Scan Food", "History"],
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.header("Welcome to Smart Food Analyzer")

    history = get_history()

    total_scans = len(history)

    if not history.empty:

        very_healthy = int(
            (history["prediction"] == "Very Healthy").sum()
        )

        healthy = int(
            (history["prediction"] == "Healthy").sum()
        )

        moderate = int(
            (history["prediction"] == "Moderate").sum()
        )

        poor = int(
            (history["prediction"] == "Poor").sum()
        )

        very_poor = int(
            (history["prediction"] == "Very Poor").sum()
        )

    else:
        very_healthy = healthy = moderate = poor = very_poor = 0

    # Dashboard metrics

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Scans", total_scans)

    c2.metric(
        "Very Healthy",
        very_healthy,
    )

    c3.metric(
        "Healthy",
        healthy,
    )

    c4.metric(
        "Poor Choices",
        poor + very_poor,
    )

    st.divider()

    # Scan history chart

    st.subheader("📊 Your Food Scanning Summary")

    if not history.empty:

        category_counts = (
            history["prediction"]
            .value_counts()
            .reindex(
                [
                    "Very Healthy",
                    "Healthy",
                    "Moderate",
                    "Poor",
                    "Very Poor",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Number of Products",
        ]

        fig = px.pie(
            category_counts,
            names="Category",
            values="Number of Products",
            title="Your Scanned Food Categories",
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Recent Scans")

        st.dataframe(
            history[
                ["product_name", "prediction", "scanned_at"]
            ].head(5),
            hide_index=True,
            use_container_width=True,
        )

        # Personalized dashboard recommendation

        healthy_count = very_healthy + healthy
        poor_count = poor + very_poor

        if healthy_count > poor_count:

            st.success(
                "Great work! Most of your scanned products "
                "are in the Healthy or Very Healthy categories."
            )

            st.info(
                "Continue comparing nutrition labels and "
                "maintaining a varied, balanced diet."
            )

        else:

            st.info(
                "Try comparing nutrition labels before choosing "
                "products. Look for options with more fibre and "
                "protein and less added sugar and sodium."
            )

    else:

        st.info(
            "You have not scanned any products yet. "
            "Scan your first product to receive dashboard suggestions."
        )

    st.divider()

    # Branded product examples

    st.subheader("⭐ Product Examples to Explore")

    st.caption(
        "These are examples to compare, not guaranteed health "
        "ratings. Check the current nutrition label and ingredients "
        "before choosing a product."
    )

    recommended_products = [
        ("Quaker", "Plain oats"),
        ("Saffola", "Plain oats"),
        ("Tata Sampann", "Roasted chana"),
        ("Yoga Bar", "Products with no added sugar"),
        ("Amul", "Plain dahi"),
        ("Epigamia", "Plain Greek yogurt"),
    ]

    product_columns = st.columns(3)

    for index, (brand, product) in enumerate(recommended_products):

        with product_columns[index % 3]:

            st.markdown(f"**{brand}**")
            st.write(product)


# ==================================================
# SCAN FOOD PAGE
# ==================================================

elif page == "Scan Food":

    st.header("📷 Scan Nutrition Label")

    source = st.radio(
        "Image source",
        ["Upload a photo", "Use camera"],
        horizontal=True,
    )

    if source == "Upload a photo":

        uploaded = st.file_uploader(
            "Upload the back-side nutrition label",
            type=["jpg", "jpeg", "png"],
        )

    else:

        uploaded = st.camera_input(
            "Take a photo of the nutrition label"
        )

    if uploaded is not None:

        image_bytes = uploaded.getvalue()

        image_id = hashlib.sha256(image_bytes).hexdigest()

        st.image(
            image_bytes,
            caption="Uploaded nutrition label",
            width=480,
        )

        if st.session_state.get("image_id") != image_id:

            st.session_state.image_id = image_id
            st.session_state.extracted = None
            st.session_state.ocr_text = ""

        # OCR button

        if st.button(
            "Read nutrition label with OCR",
            type="primary",
        ):

            try:

                text = extract_text(image_bytes)

                st.session_state.ocr_text = text

                st.session_state.extracted = extract_nutrition(text)

                if not text.strip():

                    st.warning(
                        "No text was detected. Enter the values manually."
                    )

                elif all(
                    value is None
                    for value in st.session_state.extracted.values()
                ):

                    st.warning(
                        "Text was detected, but nutrition values "
                        "were not recognized. Enter them manually."
                    )

                else:

                    st.success(
                        "OCR completed. Please verify every value and unit."
                    )

            except Exception as exc:

                st.error(f"OCR failed: {exc}")

                st.session_state.extracted = {
                    feature: None for feature in FEATURES
                }

        # CNN estimation button

        if CNN_PATH.is_file() and CNN_SCALER_PATH.is_file():

            if st.button("Generate CNN Nutrition Estimates"):

                try:

                    st.session_state.extracted = predict_cnn_values(
                        image_bytes
                    )

                    st.success(
                        "CNN estimates generated. Verify every value "
                        "against the printed label."
                    )

                except Exception as exc:

                    st.error(f"CNN prediction failed: {exc}")

        else:

            st.caption(
                "CNN estimates are disabled because trained CNN "
                "weights are not present."
            )

        # Nutrition input form

        if st.session_state.get("extracted") is not None:

            if st.session_state.ocr_text:

                with st.expander("View recognized text"):

                    st.text(st.session_state.ocr_text)

            st.subheader("Verify / Correct Nutrition Values")

            st.info(
                "Enter values exactly as printed on the package. "
                "The app converts per-serving values into values "
                "per 100 g/ml."
            )

            image_key = st.session_state.image_id

            basis = st.radio(
                "Values shown on package",
                ["Per 100 g/ml", "Per serving"],
                horizontal=True,
                key=f"basis_{image_key}",
            )

            serving_size = None

            if basis == "Per serving":

                serving_size = st.number_input(
                    "Serving size (g or ml)",
                    min_value=0.1,
                    value=30.0,
                    step=0.1,
                    key=f"serving_{image_key}",
                )

            with st.form(f"nutrition_form_{image_key}"):

                product_name = st.text_input("Product name")

                price = st.number_input(
                    "Product price (₹)",
                    min_value=0.0,
                    value=50.0,
                    step=1.0,
                )

                entered = {}

                columns = st.columns(2)

                for index, feature in enumerate(FEATURES):

                    default = st.session_state.extracted.get(feature)

                    with columns[index % 2]:

                        entered[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()} "
                            f"({UNITS[feature]})",
                            min_value=0.0,
                            value=(
                                float(default)
                                if default is not None
                                else 0.0
                            ),
                            step=0.1,
                            format="%.2f",
                            help=(
                                "If OCR did not find this value, "
                                "replace 0 with the value printed "
                                "on the label."
                            ),
                        )

                ingredients = st.text_area(
                    "Ingredients (optional)"
                )

                submitted = st.form_submit_button(
                    "Analyze with Fuzzy Logic",
                    type="primary",
                )

            # Analysis

            if submitted:

                if basis == "Per serving" and (
                    serving_size is None or serving_size <= 0
                ):

                    st.error(
                        "Serving size must be greater than zero."
                    )

                else:

                    factor = (
                        1.0
                        if basis == "Per 100 g/ml"
                        else 100.0 / serving_size
                    )

                    per_100 = {
                        feature: float(entered[feature]) * factor
                        for feature in FEATURES
                    }

                    # Basic nutrition value checks

                    if (
                        per_100["sodium"] > 10000
                        or per_100["calories"] > 1000
                        or any(
                            per_100[key] > 100
                            for key in [
                                "sugar",
                                "protein",
                                "fat",
                                "carbohydrates",
                                "fiber",
                            ]
                        )
                    ):

                        st.error(
                            "At least one value looks unusually high "
                            "per 100 g/ml. Check the numbers and units."
                        )

                    else:

                        # Fuzzy logic calculation

                        health_score, prediction = calculate_health_score(
                            per_100
                        )

                        st.divider()

                        st.subheader(
                            "🥗 Food Health Analysis (per 100 g/ml)"
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            show_health_meter(
                                health_score,
                                prediction,
                            )

                        with col2:

                            st.metric(
                                "Health Category",
                                prediction,
                            )

                            st.metric(
                                "Health Score",
                                f"{health_score:.2f}%",
                            )

                            show_category_message(prediction)

                        # Nutrition table

                        st.subheader("Nutrition Values")

                        nutrition_table = pd.DataFrame({
                            "Nutrient": FEATURES,
                            "Value per 100 g/ml": [
                                round(per_100[feature], 2)
                                for feature in FEATURES
                            ],
                            "Unit": [
                                UNITS[feature]
                                for feature in FEATURES
                            ],
                        })

                        st.dataframe(
                            nutrition_table,
                            hide_index=True,
                            use_container_width=True,
                        )

                        # Save scan to database

                        save_scan({
                            "name": (
                                product_name.strip()
                                or "Unnamed Product"
                            ),
                            "price": price,
                            **per_100,
                            "prediction": prediction,
                            "confidence": None,
                            "ingredients": ingredients,
                        })

                        st.success(
                            "Your food scan has been saved successfully!"
                        )

                        # Alternative product suggestions

                        if prediction not in [
                            "Very Healthy",
                            "Healthy",
                        ]:

                            alternatives = get_alternatives(price)

                            st.subheader(
                                "🌱 Possible Healthier Alternatives"
                            )

                            if alternatives.empty:

                                st.info(
                                    "No seeded alternatives were found "
                                    "at or below this price."
                                )

                            else:

                                st.dataframe(
                                    alternatives,
                                    hide_index=True,
                                    use_container_width=True,
                                )

    else:

        st.info(
            "Upload or capture a clear photo of the nutrition "
            "information panel to begin."
        )


# ==================================================
# HISTORY PAGE
# ==================================================

else:

    st.header("📊 Scan History")

    history = get_history()

    if history.empty:

        st.info("You have not scanned any products yet.")

    else:

        st.subheader("All Scanned Products")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Food Category Distribution")

        category_order = [
            "Very Healthy",
            "Healthy",
            "Moderate",
            "Poor",
            "Very Poor",
        ]

        category_counts = (
            history["prediction"]
            .value_counts()
            .reindex(category_order, fill_value=0)
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Number of Products",
        ]

        fig = px.pie(
            category_counts,
            names="Category",
            values="Number of Products",
            title="Distribution of Food Health Categories",
            hole=0.45,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
