import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    :root{--bg:#0b1220;--panel:#071021;--accent:#06b6d4;--muted:#98a8b9}
    html,body,main{background:linear-gradient(180deg,#071021 0%, #0b1220 100%);color:#e6eef6;font-family:Inter,Segoe UI,Helvetica,Arial}
    .banner{background:linear-gradient(90deg, rgba(6,182,212,0.08), rgba(99,102,241,0.04));padding:18px;border-radius:12px;margin-bottom:18px}
    .title{font-size:2rem;font-weight:700;margin:0}
    .subtitle{color:var(--muted);margin:0}
    .card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));padding:12px;border-radius:10px;margin-bottom:12px}
    .metric{font-size:1.4rem;font-weight:700}
    .download-btn{background:var(--accent);color:#071021;padding:8px 12px;border-radius:8px;text-decoration:none}
    .stButton>button {background: linear-gradient(90deg,#06b6d4,#6366f1);color:#021}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Utilities ---
MODEL_DIR = "model"
RAW_TESTDATA_URL = "https://raw.githubusercontent.com/sreelathakandula/ML/main/testdata.csv"

@st.cache_resource
def load_models_and_artifacts():
    models = {}
    scaler = None
    label_encoders = None
    feature_names = None

    # expected model files
    model_mapping = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
    }

    for name, fname in model_mapping.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                # skip if cannot load
                pass

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None
    if os.path.exists(encoders_path):
        try:
            label_encoders = joblib.load(encoders_path)
        except Exception:
            label_encoders = None
    if os.path.exists(features_path):
        try:
            feature_names = joblib.load(features_path)
        except Exception:
            feature_names = None

    return models, scaler, label_encoders, feature_names

@st.cache_data
def load_model_results():
    path = os.path.join(MODEL_DIR, 'model_comparison_results.csv')
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0)
            return df
        except Exception:
            return None
    return None

@st.cache_data(ttl=3600)
def fetch_testdata_bytes(url: str):
    """Fetch bytes for the sample testdata CSV. Cached for 1 hour."""
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception:
        return None


def preprocess(df, label_encoders, scaler, feature_names):
    df = df.copy()
    # try to handle separators and stray target
    if 'y' in df.columns:
        y = df['y'].map({'yes':1,'no':0}).fillna(df['y'])
        df = df.drop(columns=['y'])
    else:
        y = None

    # encode categorical
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        if label_encoders and col in label_encoders:
            le = label_encoders[col]
            def map_label(x):
                try:
                    return le.transform([x])[0]
                except Exception:
                    return -1
            df[col] = df[col].astype(str).apply(map_label)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        for m in missing:
            df[m] = 0
        df = df[feature_names]

    X = df.values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass
    return X, y


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    try:
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        if y_prob is not None:
            try:
                if y_prob.ndim>1 and y_prob.shape[1]>1:
                    metrics['AUC'] = roc_auc_score(y_true, y_prob[:,1])
                else:
                    metrics['AUC'] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics['AUC'] = 0.0
        else:
            metrics['AUC'] = 0.0
    except Exception:
        pass
    return metrics


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

# --- App layout ---

def main():
    st.markdown(f"""
    <div class='banner'>
      <div style='display:flex;justify-content:space-between;align-items:center'>
        <div>
          <div class='title'>🏦 Bank Marketing Classifier</div>
          <div class='subtitle'>Multi-model comparison & prediction interface</div>
        </div>
        <div id='download-slot'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Top download removed: sample CSV download is available in the Upload section below
    st.write("")

    models, scaler, label_encoders, feature_names = load_models_and_artifacts()
    model_results = load_model_results()

    # Short project summary shown at the start
    st.markdown("""
    <div class="card">
      <strong>Project summary:</strong> This application trains and compares six classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost) on the Bank Marketing dataset. Use the UI below to upload test data, choose a model (automatically selects XGBoost after upload), view evaluation metrics, and download predictions.
    </div>
    """, unsafe_allow_html=True)

    # --- Overview Section moved up (shown before model selection) ---
    st.markdown('<div class="card">Quickly upload test data, select a trained model and run predictions.</div>', unsafe_allow_html=True)
    col1_preview, col2_preview = st.columns([2,1])
    with col1_preview:
        st.header('Overview')
        st.write('Use the controls below to pick a model and the sections further down to run predictions and view comparisons.')
    with col2_preview:
        if model_results is not None:
            acc = model_results['Accuracy'].max() if 'Accuracy' in model_results.columns else None
            auc = model_results['AUC'].max() if 'AUC' in model_results.columns else None
            best = model_results['F1 Score'].idxmax() if 'F1 Score' in model_results.columns else None
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric('Top Accuracy', f"{acc:.4f}" if acc is not None else 'N/A')
            mc2.metric('Top AUC', f"{auc:.4f}" if auc is not None else 'N/A')
            mc3.metric('Best (F1)', f"{best}" if best is not None else 'N/A')
        else:
            st.info('Model comparison results not found. Run model training to generate results.')

    st.markdown('---')

    # Top controls (on-page)
    model_names = list(models.keys())
    if model_names:
        # Use session_state so we can programmatically update the selection after upload
        default_index = len(model_names)-1
        if 'selected_model' in st.session_state and st.session_state['selected_model'] in model_names:
            selected_index = model_names.index(st.session_state['selected_model'])
        else:
            selected_index = default_index
        st.selectbox('Select model to use for predictions', model_names, index=selected_index, key='selected_model')
    else:
        st.write('No trained models found in the model/ directory.')
        if 'selected_model' in st.session_state:
            del st.session_state['selected_model']

    st.markdown('---')

    # --- Upload & Predict Section ---
    st.header('Upload CSV & Predict')
    # Download sample test data placed next to the upload area for convenience
    dl_col_left, dl_col_right = st.columns([8,2])
    with dl_col_left:
        st.write("")
    with dl_col_right:
        sample_bytes_small = fetch_testdata_bytes(RAW_TESTDATA_URL)
        if sample_bytes_small:
            st.download_button(
                label="📥 Download testdata.csv",
                data=sample_bytes_small,
                file_name="testdata.csv",
                mime="text/csv",
                key="download_upload",
            )
        else:
            st.markdown(f"[📥 Download testdata.csv]({RAW_TESTDATA_URL})")
    uploaded = st.file_uploader('Upload CSV file (semicolon or comma separated)', type=['csv'])
    # Determine an auto-selected model (prefer XGBoost on upload) without writing to session_state
    auto_selected_model = None
    if uploaded is not None and 'XGBoost' in model_names:
        auto_selected_model = 'XGBoost'

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, sep=';')
            if df.shape[1]==1:
                uploaded.seek(0)
                df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)

        st.write('Preview:')
        st.dataframe(df.head())

        has_target = 'y' in df.columns

        # Determine effective model to use: prefer auto_selected_model (XGBoost on upload), else session_state selection
        current_model = auto_selected_model if auto_selected_model is not None else st.session_state.get('selected_model', None)
        if current_model is None:
            st.error('No trained models found in model/ directory. Please train or add models.')
        else:
            if st.button('Run Predictions'):
                X, y_true = preprocess(df, label_encoders, scaler, feature_names)
                model = models.get(current_model)
                if model is None:
                    st.error('Selected model not available.')
                else:
                    y_pred = model.predict(X)
                    try:
                        y_prob = model.predict_proba(X)
                    except Exception:
                        y_prob = None

                    st.success('Predictions done')
                    st.write('Prediction counts:')
                    st.write(pd.Series(y_pred).value_counts())

                    if has_target and y_true is not None:
                        metrics = calculate_metrics(y_true, y_pred, y_prob)
                        st.subheader('Evaluation Metrics')
                        cols = st.columns(3)
                        cols[0].metric('Accuracy', f"{metrics.get('Accuracy',0):.4f}")
                        cols[1].metric('AUC', f"{metrics.get('AUC',0):.4f}")
                        cols[2].metric('F1', f"{metrics.get('F1',0):.4f}")

                        st.subheader('Confusion Matrix')
                        fig = plot_confusion(y_true, y_pred)
                        st.pyplot(fig)

                        st.subheader('Classification Report')
                        report = classification_report(y_true, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())

                    # download results
                    out = df.copy()
                    out['Prediction'] = ['Yes' if p==1 else 'No' for p in y_pred]
                    if y_prob is not None:
                        try:
                            out['Prob_Yes'] = y_prob[:,1]
                        except Exception:
                            out['Prob'] = list(y_prob)
                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button('Download predictions', data=csv, file_name='predictions.csv', mime='text/csv', key='download_predictions')

    st.markdown('---')

    # --- Model Comparison Section ---
    st.header('Model Comparison')
    if model_results is not None:
        st.dataframe(model_results)
        st.subheader('Metric comparison')
        metrics = ['Accuracy','AUC','Precision','Recall','F1 Score','MCC']
        available = [m for m in metrics if m in model_results.columns]
        if available:
            fig,ax = plt.subplots(figsize=(10,4))
            model_results[available].plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.info('No model comparison file found. Run ML_assignment.py to generate results in model/model_comparison_results.csv')

    st.markdown('---')

    # --- About Section ---
    st.header('About')
    st.markdown('''
    This Streamlit app demonstrates multiple classification models trained on the Bank Marketing dataset.
    
    Features:
    - Upload CSV and make predictions
    - Select from multiple trained models
    - View evaluation metrics, confusion matrix and classification report
    ''')

    # footer (on-page)
    st.markdown('---')
    st.write('📅 Date: February 2026')

if __name__=='__main__':
    main()

