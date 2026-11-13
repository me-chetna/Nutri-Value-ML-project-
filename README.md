# Nutri-Value: Smart Food Health Analyzer

## 1. Project Overview

Nutri-Value is an AI-based food nutrition analysis application that helps users understand the nutritional quality of packaged food products.

The project combines **Deep Learning, Optical Character Recognition (OCR), and Soft Computing (Fuzzy Logic)** to extract nutritional information and calculate a food health score.

The application analyzes seven nutritional parameters:

- Calories
- Sugar
- Protein
- Fat
- Sodium
- Carbohydrates
- Fibre

The final health score ranges from 0 to 100 and classifies food into five categories.

## 2. Technologies and Concepts Used

### A. Deep Learning – Convolutional Neural Network (CNN)

CNN is a deep-learning architecture designed for processing images and learning visual features automatically.

Our project implements a CNN regression model called `FoodNutritionCNN`.

**CNN Architecture:**

| Layer | Filters / Units | Purpose |
|---|---:|---|
| Conv2D | 32 | Extracts basic image features |
| Conv2D | 64 | Learns more complex features |
| Conv2D | 128 | Extracts higher-level features |
| Conv2D | 192 | Learns deeper visual patterns |
| ReLU | — | Introduces non-linearity |
| Batch Normalization | — | Stabilizes training |
| Max Pooling | 2 × 2 | Reduces image dimensions |
| Adaptive Average Pooling | 4 × 4 | Produces fixed-size features |
| Flatten | — | Converts features into a vector |
| Linear | 256 | Learns feature relationships |
| Dropout | 0.25 | Helps reduce overfitting |
| Output Layer | 7 | Predicts seven nutritional values |

**Important Deep Learning concepts:**

- Convolution: Extracts visual features.
- Activation Function: ReLU introduces non-linearity.
- Pooling: Reduces spatial dimensions.
- Backpropagation: Updates model weights using prediction errors.
- Loss Function: SmoothL1Loss measures prediction error.
- Optimizer: AdamW updates model parameters during training.

The CNN requires labelled nutrition-label images and their corresponding seven nutritional values for training.

### B. Optical Character Recognition (OCR)

OCR converts text present in an image into machine-readable text.

We use EasyOCR to read printed nutritional information from food packaging.

**Working:**

1. User uploads a food nutrition-label image.
2. The image is opened and converted into RGB format.
3. Small images are enlarged to improve text recognition.
4. EasyOCR detects and recognizes text.
5. Regular Expressions identify nutritional values.
6. Extracted values are displayed for user verification.

OCR helps extract actual printed nutritional information without manually entering every value.

### C. Soft Computing – Fuzzy Logic

Fuzzy Logic is a soft-computing technique that handles uncertainty and approximate reasoning.

Unlike traditional binary logic, fuzzy logic allows partial membership between 0 and 1.

Our project uses two membership functions:

**1. Triangular Membership Function**

Defined by three parameters: a, b and c.

- a: Starting point
- b: Peak point
- c: Ending point

It represents a category with one peak.

**2. Trapezoidal Membership Function**

Defined by four parameters: a, b, c and d.

It contains a flat region where membership remains equal to 1.

**Why do we use both?**

Triangular functions represent moderate nutrient ranges, while trapezoidal functions represent ranges that can have full membership.

**Nutritional classification:**

Lower values are generally preferred for:
- Calories
- Sugar
- Fat
- Sodium
- Carbohydrates

Higher values are generally preferred for:
- Protein
- Fibre

**Weighted Membership Calculation:**

Different nutrients have different importance weights.

| Nutrient | Weight |
|---|---:|
| Calories | 1.0 |
| Sugar | 1.5 |
| Protein | 1.0 |
| Fat | 1.0 |
| Sodium | 1.5 |
| Carbohydrates | 0.75 |
| Fibre | 1.25 |

**Defuzzification:**

The project converts fuzzy memberships into a numerical health score using weighted average calculation.

Output values:
- Good = 95
- Moderate = 60
- Poor = 15

Final score:

Score = Σ(Membership × Output Value) / Σ(Membership)

**Health Categories:**

| Score | Category |
|---|---|
| 80–100 | Very Healthy |
| 65–79 | Healthy |
| 45–64 | Moderate |
| 25–44 | Poor |
| Below 25 | Very Poor |

These are project-defined nutritional screening rules, not medical standards.

## 3. System Workflow

1. User uploads a packaged-food nutrition-label image.
2. OCR extracts printed nutritional information.
3. Regular Expressions identify the seven nutritional values.
4. User verifies and corrects the extracted values.
5. Values are normalized to per 100 g/ml.
6. Fuzzy Logic calculates membership values.
7. Weighted defuzzification generates the final health score.
8. The application displays the health category.
9. Scan information is stored in the SQLite database.

**CNN Module:** The CNN is a separate experimental image-regression module designed to estimate nutritional values directly from images. It requires a trained model and scaler before predictions can be generated.

## 4. Project Structure

```text
Nutri-Value-ML-project/
│
├── app.py
├── cnn_model.py
├── train_cnn.py
├── ocr_utils.py
├── fuzzy_health.py
├── database.py
│
├── nutrition.csv
├── requirements.txt
├── README.md
│
├── models/
│   ├── cnn_nutrition_model.pth
│   └── cnn_target_scaler.pkl
│
├── tests/
│   └── test_ocr_utils.py
│
└── food_scans.db
```

The CNN model and scaler files are generated after successful training.

## 5. Libraries Used

| Library | Application |
|---|---|
| Python | Main programming language |
| PyTorch | CNN implementation and training |
| Torchvision | Image preprocessing |
| EasyOCR | Text recognition |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Pillow | Image handling |
| Scikit-learn | StandardScaler for CNN targets |
| Joblib | Saving and loading the scaler |
| Streamlit | Web application interface |
| Plotly | Interactive graphs |
| SQLite3 | Database management |
| Regex | Nutritional value extraction |

## 6. Project Initialization and Installation

### Step 1: Install Python

Install Python 3.10 or 3.11 and verify installation:

```bash
python --version
```

### Step 2: Extract the Project

Extract the project ZIP file and open the project folder in VS Code.

### Step 3: Open Terminal

Navigate to the project directory:

```bash
cd Nutri-Value-ML-project-
```

### Step 4: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

Default local address:

`http://localhost:8501`

**Note:** EasyOCR may download its pretrained recognition models during the first scan. An internet connection may be required for this initial download.

## 7. CNN Training Instructions

The CNN training script is `train_cnn.py`.

Before training, prepare a CSV file named `cnn_dataset.csv`.

Required columns:

```text
image_path
calories
sugar
protein
fat
sodium
carbohydrates
fiber
```

All nutritional values must correspond to the image and use a consistent per-100-g/ml basis.

Place the corresponding images in the project folder.

Example:

```text
Nutri-Value-ML-project/
│
├── cnn_dataset.csv
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
```

Start training:

```bash
python train_cnn.py
```

The training script:

- Resizes images to 224 × 224.
- Normalizes image inputs.
- Scales nutritional targets using StandardScaler.
- Splits data into training and validation sets.
- Uses AdamW optimizer.
- Uses SmoothL1Loss.
- Trains for 25 epochs.
- Saves the best model and target scaler in the models folder.

At least 20 labelled images are required by the current script to start training. However, several hundred diverse labelled images or more are recommended for meaningful results.

A separate test dataset should be used to evaluate model performance before relying on predictions.

## 8. Database

The application uses SQLite to store food scan history.

The database is initialized automatically when the application starts.

It stores:
- Product name
- Price
- Nutritional information
- Health category
- Scan date and time

No separate database server is required.

## 9. How to Use the Application

1. Start the Streamlit application.
2. Navigate to the Scan Food section.
3. Upload a food nutrition-label image or use the camera.
4. Click Read Nutrition Label with OCR.
5. Verify all extracted nutritional values.
6. Enter or correct missing values.
7. Select whether values are per serving or per 100 g/ml.
8. Enter the product information.
9. Generate the health analysis.
10. View the health score and category.
11. Check previous scans in the History section.

## 10. Advantages

- Automates nutritional text extraction.
- Combines deep learning and soft computing concepts.
- Uses fuzzy membership functions to handle approximate nutritional classifications.
- Provides an understandable health score.
- Maintains previous scan records.
- Offers an interactive dashboard.

## 11. Limitations

- OCR accuracy depends on image quality.
- Extracted values must be verified by the user.
- CNN predictions depend on the quality and quantity of labelled training images.
- Fuzzy thresholds are approximate project-defined rules.
- The health score is an informational screening tool, not medical advice.

## 12. Future Scope

- Train CNN using a larger nutrition-label image dataset.
- Improve OCR accuracy and multilingual support.
- Integrate verified nutritional databases.
- Improve food recommendation functionality.
- Evaluate CNN performance using independent test data.
- Extend the application to mobile platforms.

## Conclusion

Nutri-Value demonstrates the practical application of Deep Learning, Computer Vision, OCR and Soft Computing in food nutrition analysis.

CNN provides an experimental approach to image-based nutritional estimation, OCR extracts printed nutritional information, and Fuzzy Logic evaluates nutritional quality through membership functions and weighted defuzzification.

The project aims to make packaged-food nutritional information easier to understand through an interactive AI-based application.
