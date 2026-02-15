import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# ── Cache: load data, train all models once ──────────────────────────────────
@st.cache_resource
def load_and_train():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        trained[name] = model
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC Score': roc_auc_score(y_test, y_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'MCC Score': matthews_corrcoef(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
        }

    return cancer, df, scaler, X_test, y_test, trained, results

cancer, df, scaler, X_test, y_test, trained_models, results = load_and_train()
feature_names = cancer.feature_names.tolist()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Model Performance",
    "Confusion Matrices",
    "Make a Prediction",
])

# ── Page: Home ───────────────────────────────────────────────────────────────
if page == "Home":
    st.title("Breast Cancer Prediction App")
    st.markdown(
        "This application trains **6 classification models** on the "
        "**Breast Cancer Wisconsin (Diagnostic)** dataset and lets you "
        "explore their performance or make live predictions."
    )

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Classes", "Malignant (0) / Benign (1)")

    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(4, 3))
    df['target'].value_counts().plot.bar(
        ax=ax, color=['#e74c3c', '#2ecc71']
    )
    ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ── Page: Model Performance ──────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    # Build metrics table (exclude prediction arrays)
    metric_cols = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score']
    metrics_df = pd.DataFrame(
        {name: {m: results[name][m] for m in metric_cols} for name in results}
    ).T

    st.subheader("Metrics Table")
    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
        use_container_width=True,
    )

    st.subheader("Bar Charts")
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = sns.color_palette("viridis", len(metrics_df))
        bars = ax.bar(metrics_df.index, metrics_df[metric], color=colors)
        ax.set_title(f'{metric} Across Models', fontsize=14)
        ax.set_ylabel(metric)
        ax.set_ylim(metrics_df[metric].min() - 0.02, 1.0)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# ── Page: Confusion Matrices ─────────────────────────────────────────────────
elif page == "Confusion Matrices":
    st.title("Confusion Matrices")

    cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Actual 0', 'Actual 1'],
        )
        ax.set_title(name, fontsize=12)
        cols[idx % 3].pyplot(fig)

# ── Page: Make a Prediction ──────────────────────────────────────────────────
elif page == "Make a Prediction":
    st.title("Make a Prediction")
    st.markdown("Adjust the feature sliders below and click **Predict**.")

    selected_model = st.selectbox("Choose a model", list(trained_models.keys()))

    # Build sliders grouped by feature category
    input_values = []
    groups = [
        ("Mean Features", feature_names[:10]),
        ("Error Features", feature_names[10:20]),
        ("Worst Features", feature_names[20:30]),
    ]

    for group_name, features in groups:
        st.subheader(group_name)
        group_cols = st.columns(5)
        for i, feat in enumerate(features):
            col_data = df[feat]
            val = group_cols[i % 5].slider(
                feat,
                float(col_data.min()),
                float(col_data.max()),
                float(col_data.mean()),
                key=feat,
            )
            input_values.append(val)

    if st.button("Predict", type="primary"):
        input_arr = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        model = trained_models[selected_model]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        if prediction == 1:
            col1.success("**Prediction: Benign**")
        else:
            col1.error("**Prediction: Malignant**")

        col2.metric("Confidence", f"{max(probability) * 100:.1f}%")

        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            'Class': ['Malignant (0)', 'Benign (1)'],
            'Probability': probability,
        })
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.barh(prob_df['Class'], prob_df['Probability'], color=['#e74c3c', '#2ecc71'])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        plt.tight_layout()
        st.pyplot(fig)
