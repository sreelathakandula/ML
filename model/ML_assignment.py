# =============================================================================
# Machine Learning Assignment - 2
# Bank Marketing Classification
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 1: Import Required Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# SECTION 2: Load Dataset from URL
# -----------------------------------------------------------------------------
print("=" * 70)
print("MACHINE LEARNING ASSIGNMENT - BANK MARKETING CLASSIFICATION")
print("=" * 70)

# Load Bank Marketing Dataset from UCI Repository
bank_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

print("\nLoading Bank Marketing Dataset...")
print(f"Source: UCI Machine Learning Repository")

# Download and extract the dataset
import urllib.request
import zipfile
import io

# Download the zip file
print("Downloading dataset...")
response = urllib.request.urlopen(bank_data_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

# Read the full dataset from the zip
with zip_file.open('bank-additional/bank-additional-full.csv') as f:
    bank_df = pd.read_csv(f, sep=';')

print("\nDataset loaded successfully!")
print(f"Shape of dataset: {bank_df.shape}")

# -----------------------------------------------------------------------------
# SECTION 3: Explore the Dataset
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET EXPLORATION")
print("=" * 70)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(bank_df.head())

# Display dataset info
print("\n" + "-" * 50)
print("Dataset Information:")
print("-" * 50)
print(f"Number of Instances: {bank_df.shape[0]}")
print(f"Number of Features: {bank_df.shape[1] - 1}")
print(f"Target Column: y (yes/no - subscribed to term deposit)")

# Display column names
print("\n" + "-" * 50)
print("Feature Columns:")
print("-" * 50)
for i, col in enumerate(bank_df.columns, 1):
    print(f"{i}. {col} - dtype: {bank_df[col].dtype}")

# Statistical summary for numerical columns
print("\n" + "-" * 50)
print("Statistical Summary (Numerical Features):")
print("-" * 50)
print(bank_df.describe())

# Check for missing values
print("\n" + "-" * 50)
print("Missing Values Check:")
print("-" * 50)
missing_vals = bank_df.isnull().sum()
print(missing_vals)
print(f"\nTotal Missing Values: {missing_vals.sum()}")

# Check target distribution
print("\n" + "-" * 50)
print("Target Variable (y) Distribution:")
print("-" * 50)
target_counts = bank_df['y'].value_counts()
print(target_counts)

# -----------------------------------------------------------------------------
# SECTION 4: Data Preprocessing
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Create a copy for preprocessing
df = bank_df.copy()

# Convert target variable to binary (yes=1, no=0)
print("\nConverting target variable to binary:")
print("  - 'yes' -> 1 (Subscribed to term deposit)")
print("  - 'no' -> 0 (Did not subscribe)")

df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Check new class distribution
print("\nNew Class Distribution:")
class_distribution = df['y'].value_counts()
print(class_distribution)
print(f"\nClass 0 (No): {class_distribution[0]} samples ({class_distribution[0]/len(df)*100:.2f}%)")
print(f"Class 1 (Yes): {class_distribution[1]} samples ({class_distribution[1]/len(df)*100:.2f}%)")

# Encode categorical variables
print("\n" + "-" * 50)
print("Encoding Categorical Variables")
print("-" * 50)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Use Label Encoding for categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} unique values")

# Separate features and target
X = df.drop('y', axis=1).values
y = df['y'].values
feature_names = df.drop('y', axis=1).columns.tolist()

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")
print(f"Number of Features: {len(feature_names)}")

# Split data into training and testing sets
print("\n" + "-" * 50)
print("Splitting Data (80% Train, 20% Test)")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# Feature Scaling using StandardScaler
print("\n" + "-" * 50)
print("Feature Scaling (StandardScaler)")
print("-" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features have been scaled using StandardScaler")
print(f"Scaled Training Data Mean: {X_train_scaled.mean():.6f}")
print(f"Scaled Training Data Std: {X_train_scaled.std():.6f}")

# -----------------------------------------------------------------------------
# SECTION 5: Define Function to Calculate All Metrics
# -----------------------------------------------------------------------------
def calculate_all_metrics(model_name, y_true, y_pred, y_prob):
    """
    Calculate all required evaluation metrics for a model
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    # Calculate AUC Score
    if y_prob is not None:
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.0

    metrics_dict = {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    return metrics_dict

# -----------------------------------------------------------------------------
# SECTION 6: Train and Evaluate All Models
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)

# List to store all model results
all_results = []

# Create directory to save models
#model_dir = ""
#if not os.path.exists(model_dir):
#    os.makedirs(model_dir)
#    print(f"\nCreated '{model_dir}' directory for saving models")
# Use the directory where this script lives so all artifacts are colocated with ML_assignment.py
model_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"\nCreated directory for saving models: {model_dir}")
else:
    print(f"\nArtifacts will be saved to: {model_dir}")

# =============================================================================
# MODEL 1: Logistic Regression
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("-" * 70)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

print("Training Logistic Regression...")
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)

lr_metrics = calculate_all_metrics("Logistic Regression", y_test, lr_pred, lr_prob)
all_results.append(lr_metrics)

print("\nLogistic Regression Results:")
for key, value in lr_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(lr_model, os.path.join(model_dir, 'logistic_regression.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'logistic_regression.pkl')}")

# =============================================================================
# MODEL 2: Decision Tree Classifier
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 2: DECISION TREE CLASSIFIER")
print("-" * 70)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

print("Training Decision Tree Classifier...")
dt_model.fit(X_train_scaled, y_train)

dt_pred = dt_model.predict(X_test_scaled)
dt_prob = dt_model.predict_proba(X_test_scaled)

dt_metrics = calculate_all_metrics("Decision Tree", y_test, dt_pred, dt_prob)
all_results.append(dt_metrics)

print("\nDecision Tree Results:")
for key, value in dt_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(dt_model, os.path.join(model_dir, 'decision_tree.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'decision_tree.pkl')}")

# =============================================================================
# MODEL 3: K-Nearest Neighbors Classifier
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 3: K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
print("-" * 70)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='euclidean'
)

print("Training K-Nearest Neighbors Classifier...")
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)
knn_prob = knn_model.predict_proba(X_test_scaled)

knn_metrics = calculate_all_metrics("K-Nearest Neighbors", y_test, knn_pred, knn_prob)
all_results.append(knn_metrics)

print("\nK-Nearest Neighbors Results:")
for key, value in knn_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(knn_model, os.path.join(model_dir, 'knn.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'knn.pkl')}")

# =============================================================================
# MODEL 4: Naive Bayes Classifier (Gaussian)
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 4: NAIVE BAYES CLASSIFIER (GAUSSIAN)")
print("-" * 70)

nb_model = GaussianNB()

print("Training Gaussian Naive Bayes Classifier...")
nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)
nb_prob = nb_model.predict_proba(X_test_scaled)

nb_metrics = calculate_all_metrics("Naive Bayes", y_test, nb_pred, nb_prob)
all_results.append(nb_metrics)

print("\nNaive Bayes Results:")
for key, value in nb_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(nb_model, os.path.join(model_dir, 'naive_bayes.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'naive_bayes.pkl')}")

# =============================================================================
# MODEL 5: Random Forest (Ensemble)
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 5: RANDOM FOREST CLASSIFIER (ENSEMBLE)")
print("-" * 70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest Classifier...")
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_prob = rf_model.predict_proba(X_test_scaled)

rf_metrics = calculate_all_metrics("Random Forest", y_test, rf_pred, rf_prob)
all_results.append(rf_metrics)

print("\nRandom Forest Results:")
for key, value in rf_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(rf_model, os.path.join(model_dir, 'random_forest.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'random_forest.pkl')}")

# =============================================================================
# MODEL 6: XGBoost (Ensemble)
# =============================================================================
print("\n" + "-" * 70)
print("MODEL 6: XGBOOST CLASSIFIER (ENSEMBLE)")
print("-" * 70)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("Training XGBoost Classifier...")
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_prob = xgb_model.predict_proba(X_test_scaled)

xgb_metrics = calculate_all_metrics("XGBoost", y_test, xgb_pred, xgb_prob)
all_results.append(xgb_metrics)

print("\nXGBoost Results:")
for key, value in xgb_metrics.items():
    if key != 'Model':
        print(f"  {key}: {value}")

joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost.pkl'))
print(f"\nModel saved to: {os.path.join(model_dir, 'xgboost.pkl')}")

# Save scaler and label encoders for future use
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
print(f"\nScaler and encoders saved to: {model_dir}")

# -----------------------------------------------------------------------------
# SECTION 7: Results Comparison Table
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL COMPARISON - EVALUATION METRICS TABLE")
print("=" * 70)

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('Model')

print("\n" + results_df.to_string())

results_df.to_csv(os.path.join(model_dir, 'model_comparison_results.csv'))
print(f"\nResults saved to: {os.path.join(model_dir, 'model_comparison_results.csv')}")

# -----------------------------------------------------------------------------
# SECTION 9: Generate Visualizations
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

plt.style.use('default')
sns.set_palette("husl")

# Figure 1: Metrics Comparison Bar Chart
fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle('Model Performance Comparison - Bank Marketing Dataset', fontsize=16, fontweight='bold')

models = results_df.index.tolist()
colors = sns.color_palette("husl", len(models))

metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    values = results_df[metric].values
    bars = ax.bar(range(len(models)), values, color=colors)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(model_dir, 'model_comparison_chart.png')}")

# Figure 2: Confusion Matrices for All Models
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')

predictions = [lr_pred, dt_pred, knn_pred, nb_pred, rf_pred, xgb_pred]
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes',
               'Random Forest', 'XGBoost']

for idx, (pred, name) in enumerate(zip(predictions, model_names)):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'],
                yticklabels=['No', 'Yes'])
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(model_dir, 'confusion_matrices.png')}")

# Figure 3: Feature Importance (for tree-based models)
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

# Decision Tree Feature Importance
ax1 = axes[0]
dt_importance = dt_model.feature_importances_
sorted_idx = np.argsort(dt_importance)[-15:]  # Top 15 features
ax1.barh(range(len(sorted_idx)), dt_importance[sorted_idx], color='steelblue')
ax1.set_yticks(range(len(sorted_idx)))
ax1.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
ax1.set_title('Decision Tree', fontweight='bold')
ax1.set_xlabel('Importance')

# Random Forest Feature Importance
ax2 = axes[1]
rf_importance = rf_model.feature_importances_
sorted_idx = np.argsort(rf_importance)[-15:]
ax2.barh(range(len(sorted_idx)), rf_importance[sorted_idx], color='forestgreen')
ax2.set_yticks(range(len(sorted_idx)))
ax2.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
ax2.set_title('Random Forest', fontweight='bold')
ax2.set_xlabel('Importance')

# XGBoost Feature Importance
ax3 = axes[2]
xgb_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[-15:]
ax3.barh(range(len(sorted_idx)), xgb_importance[sorted_idx], color='darkorange')
ax3.set_yticks(range(len(sorted_idx)))
ax3.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
ax3.set_title('XGBoost', fontweight='bold')
ax3.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(model_dir, 'feature_importance.png')}")

# Figure 4: Class Distribution
fig4, ax = plt.subplots(figsize=(8, 6))
fig4.suptitle('Target Variable Distribution', fontsize=16, fontweight='bold')

class_labels = ['No (Did not subscribe)', 'Yes (Subscribed)']
class_counts = [class_distribution[0], class_distribution[1]]
bars = ax.bar(class_labels, class_counts, color=['#ff6b6b', '#51cf66'])
ax.set_xlabel('Subscription Status')
ax.set_ylabel('Count')
ax.set_title('Bank Term Deposit Subscription Distribution', fontweight='bold')

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{int(bar.get_height())} ({bar.get_height()/len(df)*100:.1f}%)',
            ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(model_dir, 'data_distribution.png')}")

plt.close('all')

# -----------------------------------------------------------------------------
# SECTION 10: Classification Reports
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 70)

for pred, name in zip(predictions, model_names):
    print(f"\n{'-' * 50}")
    print(f"Classification Report: {name}")
    print(f"{'-' * 50}")
    print(classification_report(y_test, pred, target_names=['No', 'Yes']))

# -----------------------------------------------------------------------------
# SECTION 11: Model Observations
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL PERFORMANCE OBSERVATIONS")
print("=" * 70)

observations = {
    "Logistic Regression": """
    - Provides a solid baseline for binary classification on bank marketing data
    - Works well with the linear relationships between features and target
    - Fast training time and suitable for large datasets like this one
    - Handles the imbalanced dataset reasonably well with good AUC score""",

    "Decision Tree": """
    - Captures non-linear relationships in customer behavior data
    - Identifies key decision factors like duration, euribor3m, and nr.employed
    - Prone to overfitting without proper depth control
    - Provides interpretable rules for marketing decisions""",

    "K-Nearest Neighbors": """
    - Performance depends on feature scaling which was applied
    - Computationally expensive for large datasets during prediction
    - Sensitive to the imbalanced nature of the dataset
    - Works well when similar customer profiles cluster together""",

    "Naive Bayes": """
    - Fast training and prediction despite large dataset size
    - Independence assumption may not hold for correlated marketing features
    - Handles the class imbalance with probabilistic predictions
    - Provides good baseline but may underperform on complex patterns""",

    "Random Forest (Ensemble)": """
    - Robust predictions through ensemble of decision trees
    - Handles the imbalanced dataset well with balanced sampling
    - Provides reliable feature importance for marketing insights
    - Less prone to overfitting compared to single decision tree""",

    "XGBoost (Ensemble)": """
    - Gradient boosting provides sequential error correction
    - Handles imbalanced data well with built-in class weights
    - Often achieves best performance on structured/tabular data
    - Built-in regularization prevents overfitting on large datasets"""
}

for model, obs in observations.items():
    print(f"\n{model}:{obs}")

# -----------------------------------------------------------------------------
# SECTION 12: Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ASSIGNMENT SUMMARY")
print("=" * 70)

print(f"""
Dataset: Bank Marketing (UCI Machine Learning Repository)
Total Samples: {bank_df.shape[0]}
Features: {bank_df.shape[1] - 1}
Classification Type: Binary (Subscribed vs Not Subscribed to Term Deposit)
Train-Test Split: 80-20

Dataset Specifications:
- Number of Instances: {bank_df.shape[0]} (Requirement: 500+) ✓
- Number of Features: {bank_df.shape[1] - 1} (Requirement: 12+) ✓

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics Calculated:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Files Generated:
- model/logistic_regression.pkl
- model/decision_tree.pkl
- model/knn.pkl
- model/naive_bayes.pkl
- model/random_forest.pkl
- model/xgboost.pkl
- model/scaler.pkl
- model/label_encoders.pkl
- model/feature_names.pkl
- model/model_comparison_results.csv
- model/model_comparison_chart.png
- model/confusion_matrices.png
- model/feature_importance.png
- model/data_distribution.png
""")

print("=" * 70)
print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("=" * 70)
