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

## Model comparison (metrics)

Make a Comparison Table with the evaluation metrics calculated for all the 6 models below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|---------:|----:|---------:|------:|----:|----:|
| Logistic Regression | 0.9139 | 0.9370 | 0.9033 | 0.9139 | 0.9039 | 0.4956 |
| Decision Tree | 0.9160 | 0.8953 | 0.9124 | 0.9160 | 0.9139 | 0.5606 |
| kNN (K-Nearest Neighbors) | 0.9053 | 0.8617 | 0.8928 | 0.9053 | 0.8956 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.8864 | 0.8536 | 0.8665 | 0.4189 |
| Random Forest (Ensemble) | 0.9217 | 0.9531 | 0.9149 | 0.9217 | 0.9165 | 0.5664 |
| XGBoost (Ensemble) | 0.9216 | 0.9550 | 0.9168 | 0.9216 | 0.9185 | 0.5808 |

---

## Observations on model performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong baseline with good accuracy and AUC; fast to train and interpretable; performs well on this dataset but may miss non-linear patterns. |
| Decision Tree | Competitive accuracy and high recall; captures non-linear interactions and is interpretable, but can overfit without pruning/tuning. |
| kNN | Reasonable performance after scaling; sensitive to feature scaling and can be slow at inference for large datasets; slightly lower AUC than tree-based ensembles. |
| Naive Bayes | Fast and simple baseline; lower accuracy than other models due to independence assumption, but still useful as a quick probabilistic estimator. |
| Random Forest (Ensemble) | One of the top performers: high accuracy and AUC, robust to overfitting, provides reliable feature importance for insights. |
| XGBoost (Ensemble) | Best overall by AUC and MCC in these results; strong predictive performance with regularization and tree boosting; recommended when seeking top accuracy. |

---

## Project Structure

A concise view of the repository layout (top-level files and important folders):

```
project-root/
├─ app.py                    # Streamlit app (single-page UI)
├─ README.md                 # Project README (this file)
├─ requirements.txt          # Python dependencies (recommended)
├─ testdata.csv              # Sample test data for quick app testing
└─ model/
   ├─ ML_assignment.py       # Training & evaluation script (produces artifacts)
   ├─ logistic_regression.pkl
   ├─ decision_tree.pkl
   ├─ knn.pkl
   ├─ naive_bayes.pkl
   ├─ random_forest.pkl
   ├─ xgboost.pkl
   ├─ scaler.pkl
   ├─ label_encoders.pkl
   ├─ feature_names.pkl
   ├─ model_comparison_results.csv
   ├─ model_comparison_chart.png
   ├─ confusion_matrices.png
   ├─ feature_importance.png
   └─ data_distribution.png
```

---

## `requirements.txt`

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
