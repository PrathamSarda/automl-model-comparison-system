import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

from automl_engine import run_full_pipeline


st.set_page_config(page_title="AutoML Model Comparison System", layout="wide")

st.title("AutoML Model Comparison System")

st.write("""
Upload a CSV dataset, perform EDA, train multiple ML models automatically,
compare their performance, and make predictions.
""")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


# ------------------------------
# Cache Pipeline (IMPORTANT)
# ------------------------------
@st.cache_data
def cached_pipeline(df, target_column):
    return run_full_pipeline(df, target_column)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ------------------------------
    # Dataset Overview
    # ------------------------------
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with col2:
        st.write("Duplicate Rows:", df.duplicated().sum())
        st.write("Missing Values:", df.isnull().sum().sum())

    # ------------------------------
    # Missing Values Chart
    # ------------------------------
    st.subheader("Missing Values Analysis")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        fig, ax = plt.subplots()
        missing.sort_values().plot.barh(ax=ax)
        st.pyplot(fig)
    else:
        st.write("No missing values detected.")

    # ------------------------------
    # Target Selection
    # ------------------------------
    possible_targets = [
        col for col in df.columns
        if 1 < df[col].nunique() <= min(20, len(df) * 0.1)
    ]

    st.info("Select a target column (preferably classification).")

    target_column = st.selectbox("Select Target Column", possible_targets)

    # ------------------------------
    # Run Pipeline
    # ------------------------------
    if st.button("Run AutoML Pipeline"):

        with st.spinner("Training models..."):
            output = cached_pipeline(df, target_column)

        st.session_state.output = output

    if "output" in st.session_state:

        output = st.session_state.output

        best_model = output["best_model"]
        base_results = output["base_results"]
        tuned_results = output["tuned_results"]
        X_test = output["X_test"]
        y_test = output["y_test"]
        problem_type = output["problem_type"]

        st.subheader("Detected Problem Type")
        st.write(problem_type)

        # ------------------------------
        # Model Comparison
        # ------------------------------
        st.subheader("Model Performance")

        fig, ax = plt.subplots()
        ax.barh(base_results["Model"], base_results["ROC-AUC"])
        st.pyplot(fig)

        st.dataframe(base_results)
        st.dataframe(tuned_results)

        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        st.subheader("Confusion Matrix")

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # ------------------------------
        # ROC Curve
        # ------------------------------
        if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):

            st.subheader("ROC Curve")

            y_prob = best_model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], '--')
            ax.legend()
            st.pyplot(fig)

        # ------------------------------
        # Feature Importance (SAFE)
        # ------------------------------
        st.subheader("Feature Importance")

        try:
            model = best_model.named_steps["model"]
            preprocessor = best_model.named_steps["preprocessing"]

            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig, ax = plt.subplots()
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.invert_yaxis()
            st.pyplot(fig)

        except:
            st.write("Feature importance not available.")

        # ------------------------------
        # Prediction Interface
        # ------------------------------
        st.subheader("Make a Prediction")

        input_data = {}

        important_columns = df.drop(columns=[target_column]).columns[:5]

        for col in important_columns:

            if df[col].dtype == "object":
                input_data[col] = st.selectbox(col, df[col].dropna().unique())
            else:
                input_data[col] = st.number_input(col, float(df[col].median()))

        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):

            prediction = best_model.predict(input_df)[0]
            st.write("Prediction:", prediction)

            if hasattr(best_model, "predict_proba"):

                probs = best_model.predict_proba(input_df)[0]
                prob = max(probs)

                st.write("Probability:", round(prob, 2))

                if prob < 0.3:
                    risk = "Low"
                elif prob < 0.7:
                    risk = "Medium"
                else:
                    risk = "High"

                st.write("Risk Level:", risk)