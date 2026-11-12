# Smart Food Health Analyzer

The app reads a packaged-food nutrition label, lets you verify the extracted
values, then classifies the product with a Random Forest model.

## How a scan works

1. Upload a label photo or take one with the camera.
2. EasyOCR recognizes the printed text. Its pretrained text-recognition model
   uses deep-learning image recognition; this project does not train a custom
   CNN because it contains no labeled nutrition-label images.
3. The app extracts calories, sugar, protein, fat, sodium, carbohydrates, and
   fiber. OCR results are approximate, so verify them before analysis.
4. Choose whether the label values are per 100 g/ml or per serving. Serving
   values are normalized to 100 g/ml before prediction.
5. The Random Forest classifies the nutrition values and saves the result to
   scan history.

The separate `cnn_model.py` is only an architecture/initialization demo. It
saves random weights and is not used by the app; those weights cannot predict
nutrition values.

## Setup on Windows

Run PowerShell from the repository folder:

```powershell
Set-Location .\Nutri-Value-ML-project-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python nutrition_model.py
python -m streamlit run app.py
```

The first OCR scan may download EasyOCR's pretrained model files. Keep the
terminal open while Streamlit is running.

## Other scripts

From the project directory, these scripts locate their data and model files
relative to their own script:

```powershell
python .\generate_test_data.py
python .\bulk_test.py
python .\predict.py
python .\visualize.py
```

The Random Forest is trained on USDA food-composition data, but its Healthy /
Unhealthy labels are generated from simple sugar, fat, and sodium thresholds.
It is a demonstration classifier, not a medically validated assessment.
