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

        fig, ax = plt.subplots(figsize=(5,3))

        missing.sort_values().plot.barh(ax=ax)

        ax.set_title("Missing Values per Column")

        st.pyplot(fig)

    else:
        st.write("No missing values detected.")


    # ------------------------------
    # Target Selection
    # ------------------------------
    possible_targets = [
    col for col in df.columns
    if df[col].nunique() <= 20 and df[col].nunique() > 1
]
    st.info("Select a target column with limited unique values (like 0/1).")

    target_column = st.selectbox(
        "Select Target Column",
        possible_targets
    )


    # ------------------------------
    # Target Distribution
    # ------------------------------
    st.subheader("Target Distribution")

    fig, ax = plt.subplots(figsize=(5,3))

    df[target_column].value_counts().plot(kind="bar", ax=ax)

    ax.set_title("Class Distribution")

    st.pyplot(fig)


    # ------------------------------
    # Correlation Heatmap
    # ------------------------------
    st.subheader("Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 1:

        fig, ax = plt.subplots(figsize=(6,4))

        sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)

        st.pyplot(fig)

    st.subheader("Debug Info")

    st.write("Columns:", df.columns.tolist())
    st.write("Unique Values:", df.nunique())

    # ------------------------------
    # Run AutoML Pipeline
    # ------------------------------
    if "model_output" not in st.session_state:
        st.session_state.model_output = None


    if st.button("Run AutoML Pipeline"):

        with st.spinner("Training models..."):

            st.session_state.model_output = run_full_pipeline(
                df,
                target_column
            )


    output = st.session_state.model_output


    if output is not None:

        best_model = output["best_model"]
        base_results = output["base_results"]
        tuned_results = output["tuned_results"]
        X_test = output["X_test"]
        y_test = output["y_test"]
        problem_type = output["problem_type"]


        # ------------------------------
        # Problem Type
        # ------------------------------
        st.subheader("Detected Problem Type")

        st.write(problem_type)


        # ------------------------------
        # Model Comparison
        # ------------------------------
        st.subheader("Model Performance Comparison")

        fig, ax = plt.subplots(figsize=(6,4))

        ax.barh(base_results["Model"], base_results["ROC-AUC"])

        ax.set_xlabel("ROC-AUC Score")

        st.pyplot(fig)


        st.subheader("Base Model Results")

        st.dataframe(base_results)


        # ------------------------------
        # Tuned Results
        # ------------------------------
        st.subheader("Tuned Model Results")

        st.dataframe(tuned_results)


        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        st.subheader("Confusion Matrix")

        y_pred = best_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4,3))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)


        # ------------------------------
        # ROC Curve
        # ------------------------------
        # ROC Curve (only for binary classification and models with predict_proba)
        if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):

            st.subheader("ROC Curve")

            y_prob = best_model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)

            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(5,4))

            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

            ax.plot([0,1], [0,1], '--')

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")

            ax.legend()

            st.pyplot(fig)

        else:

            st.info("ROC curve available only for binary classification models with probability output.")


        # ------------------------------
        # Feature Importance
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
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            ).head(10)

            fig, ax = plt.subplots(figsize=(6,4))

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )

            ax.invert_yaxis()

            st.pyplot(fig)

        except:

            st.write("Feature importance not available for this model.")

        #------------------------------
        # Best Model Selection
        #------------------------------

        model = best_model.named_steps["model"]
        preprocessor = best_model.named_steps["preprocessing"]

        feature_names = preprocessor.get_feature_names_out()

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        top_features = importance_df["Feature"].head(5).tolist()

        important_columns = []

        for f in top_features:

            col = f.split("__")[-1]

            col = col.split("_")[0]

            important_columns.append(col)

        important_columns = list(set(important_columns))


        # ------------------------------
        # Prediction Interface
        # ------------------------------
        # ------------------------------

        st.subheader("Make a Prediction")

        input_data = {}

        # Identify top 5 important original columns
        important_columns = []

        for feature in importance_df["Feature"]:

            feature = feature.split("__")[-1]

            for col in df.columns:
                if feature.startswith(col):
                    important_columns.append(col)
                    break

            if len(set(important_columns)) >= 5:
                break

        important_columns = list(set(important_columns))


        # ------------------------------
        # User Input Fields
        # ------------------------------
        st.subheader("Prediction Interface")

        for column in important_columns:

            if df[column].dtype == "object":

                input_data[column] = st.selectbox(
                    column,
                    df[column].dropna().unique()
                )

            else:

                input_data[column] = st.number_input(
                    column,
                    value=float(df[column].median())
                )


        # ------------------------------
        # Create Full Input For Model
        # ------------------------------
        full_input = {}

        for col in df.columns:

            if col == target_column:
                continue

            if col in input_data:

                full_input[col] = input_data[col]

            else:

                if df[col].dtype == "object":
                    full_input[col] = df[col].mode()[0]
                else:
                    full_input[col] = float(df[col].median())


        input_df = pd.DataFrame([full_input])


        # ------------------------------
        # Prediction Button
        # ------------------------------
        if st.button("Predict"):

            prediction = best_model.predict(input_df)[0]

            st.subheader("Prediction Result")
            st.write("Prediction:", prediction)

            if hasattr(best_model, "predict_proba"):

                prob = best_model.predict_proba(input_df)[0][1]

                st.write("Fraud Probability:", round(prob,2))

                if prob < 0.3:
                    risk = "Low Risk"
                elif prob < 0.7:
                    risk = "Medium Risk"
                else:
                    risk = "High Risk"

                st.write("Risk Level:", risk)
        start = time.time()
        results = run_full_pipeline(df, target)
        st.write(f"⏱ Total time: {time.time() - start:.2f} sec")