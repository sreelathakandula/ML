# Bank Marketing Classification

A compact project that trains and compares several classification models to predict whether a bank client will subscribe to a term deposit. This repository contains the training script, trained model artifacts, and a Streamlit app for quick inference and visual inspection of results.

---

## a. Problem Statement

The objective is to build and evaluate multiple machine learning classification models on the Bank Marketing dataset to predict whether a client will subscribe to a term deposit (binary classification).

Key points:
- Task type: Binary classification (target column `y`: yes/no)
- Goal: Compare multiple models and produce reusable artifacts (models, scalers, encoders, visualizations)

---

## b. Dataset

Dataset used: Bank Marketing dataset (UCI Machine Learning Repository). A sample `testdata.csv` is included at the repository root for quick testing of the Streamlit app.

---

## c. Models Implemented

This project implements and evaluates the following models:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Random Forest
- XGBoost

Each model's trained object (pickle), preprocessing artifacts, and evaluation visualizations are saved when the training script runs.

---

## Project Structure

```
project-folder/
│-- .git/                         # Git metadata (not part of the app)
│-- app.py                        # Streamlit web application (single-page UI)
│-- README.md                     # This file
│-- requirements.txt              # (recommended) Python dependencies
│-- testdata.csv                  # Sample test data for the app
│-- model/                        # Training script and generated artifacts
│   │-- ML_assignment.py          # Training & evaluation script (runs end-to-end)
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- xgboost.pkl
│   │-- scaler.pkl
│   │-- label_encoders.pkl
│   │-- feature_names.pkl
│   │-- model_comparison_results.csv
│   │-- model_comparison_chart.png
│   │-- confusion_matrices.png
│   │-- feature_importance.png
│   │-- data_distribution.png
```

Notes:
- The `ML_assignment.py` inside the `model/` directory saves all generated `.pkl`, `.csv` and `.png` artifacts into the same `model/` directory by design.
- If you prefer artifacts in a different folder, update the `model_dir` variable at the top of `model/ML_assignment.py`.

---

## Step 4: Create `requirements.txt`

A `requirements.txt` file is recommended so your environment is reproducible and deployment is simpler.

Create `requirements.txt` at the project root (next to `app.py` and `README.md`). Example minimal contents (pin versions as needed):

```
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

To generate automatically from an active virtual environment:

```powershell
pip freeze > requirements.txt
```

To install from the file:

```powershell
pip install -r requirements.txt
```

---

## Running the training script

To reproduce the trained artifacts (models + plots), run the script located in the `model/` folder:

```powershell
python .\model\ML_assignment.py
```

After it completes, check the `model/` folder for the generated .pkl, .csv and .png files.

---

## Streamlit app (brief)

The repository includes a Streamlit front-end (`app.py`) that allows you to upload a CSV, select a trained model, run predictions, and download results. A small sample `testdata.csv` is included for quick testing.

---

If you'd like, I can also:
- Update `app.py` to use the local `testdata.csv` for the download button (so it does not reference external user-specific URLs),
- Fix the Streamlit duplicate element ID and session_state assignment errors you saw earlier, and
- Consolidate the UI into the single-page layout you requested.

Tell me which of the above you'd like me to do next and I'll apply the changes and run quick validations.
