# 🍎 Smart Food Health Analyzer

## 📌 Project Description

Smart Food Health Analyzer is a machine learning-based application that evaluates whether a food item is **healthy or unhealthy** using its **nutritional values and ingredients**.

The system combines:
- 🤖 Machine Learning (Random Forest Classifier)
- ⚠️ Ingredient Risk Detection
- 📊 Data Visualization
- 🌐 Interactive Web UI (Streamlit)

It provides both **prediction** and **explainability**, making it a complete intelligent health analysis system.

---

## 🚀 Features

- ✅ Predict food as **Healthy / Unhealthy**
- 📊 Generate a **Health Score (0–100)**
- ⚠️ Detect harmful ingredients in food
- 📦 Bulk testing using CSV datasets
- 📈 Data visualization (graphs and trends)
- 🌐 Interactive UI using Streamlit

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

---

## 📂 Project Structure
NUTRI_VAL/
│
├── app.py # Streamlit UI for user interaction
├── nutrition_model.py # Model training script
├── predict.py # Single prediction script
├── bulk_test.py # Bulk testing using CSV file
├── generate_test_data.py # Script to generate test dataset
├── visualize.py # Graphs and data visualization
│
├── nutrition.csv # Original dataset
├── test_data.csv # Generated dataset for testing
│
├── nutrition_model.pkl # Trained machine learning model
├── scaler.pkl # Feature scaler
│
├── README.md # Project documentation
└── .venv # Virtual environment


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository


git clone <your-repo-link>
cd NUTRI_VAL
---

### 2️⃣ Create Virtual Environment

#### Windows:

python -m venv .venv
..venv\Scripts\activate

#### Mac/Linux:

python3 -m venv .venv
source .venv/bin/activate
---

### 3️⃣ Install Dependencies
pip install pandas numpy scikit-learn joblib matplotlib streamlit
---

### 4️⃣ Train the Model
python nutrition_model.py

This will generate:
- `nutrition_model.pkl`
- `scaler.pkl`
---

### 5️⃣ Run the Application
streamlit run app.py
---

## 🧪 Testing the Model

### 🔹 Single Input Testing
python predict.py
---

### 🔹 Generate Test Dataset
python generate_test_data.py
---

### 🔹 Bulk Testing
python bulk_test.py
---

### 🔹 Visualization
python visualize.py
---

## 📊 Model Details

- Model: Random Forest Classifier
- Features used:
  - Calories
  - Sugar
  - Protein
  - Fat
  - Sodium
  - Carbohydrates
  - Fiber

---

## ⚠️ Ingredient Analysis

The system detects harmful ingredients such as:
- High fructose corn syrup
- Aspartame
- Monosodium glutamate
- Sodium benzoate
- Trans fat
- Artificial colors
- Palm oil
- Refined sugar

---

## 📈 Health Score Logic

The health score is calculated based on:

- Sugar (higher → worse)
- Fat (higher → worse)
- Sodium (higher → worse)
- Protein (higher → better)
- Fiber (higher → better)

### Score Interpretation:

- 🟢 70–100 → Healthy
- 🟡 40–70 → Moderate
- 🔴 0–40 → Unhealthy

---

## 🎯 Use Cases

- Nutrition awareness
- Food product analysis
- Health recommendation systems
- Educational machine learning project

---

## 💡 Future Improvements

- 📸 OCR to scan ingredients from images
- 🌐 Backend API (Flask/FastAPI)
- 📱 Mobile app integration
- 🤖 AI chatbot for food recommendations

---

## 👨‍💻 Team
Chetna Jain
Tanishq Bainivaal
Kunal Hudda
---
