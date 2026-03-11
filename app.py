import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, roc_curve, auc

from automl_engine import run_full_pipeline


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AutoML Model Comparison", layout="wide")


st.title("AutoML Model Comparison System")

st.write("""
Upload a CSV dataset, select the target column, 
and the system will automatically:

• Clean the data  
• Train multiple ML models  
• Perform hyperparameter tuning  
• Compare model performance  
• Display evaluation graphs
""")


# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")


    # -----------------------------
    # Target Column Selection
    # -----------------------------
    possible_targets = [
        col for col in df.columns 
        if df[col].nunique() <= 20
    ]

    if len(possible_targets) == 0:
        st.error("No suitable target column found.")
        st.stop()

    target_column = st.selectbox(
        "Select Target Column",
        possible_targets
    )


    # -----------------------------
    # Run Pipeline
    # -----------------------------
    if st.button("Run AutoML Pipeline"):

        with st.spinner("Running AutoML pipeline..."):

            output = run_full_pipeline(df, target_column)

        st.success("Pipeline completed!")


        # -----------------------------
        # Base Model Results
        # -----------------------------
        st.subheader("Base Model Results")

        base_results = output["base_results"]

        st.dataframe(base_results, use_container_width=True)


        # -----------------------------
        # Model Comparison Chart
        # -----------------------------
        st.subheader("Model Performance Comparison")

        if "ROC-AUC" in base_results.columns:

            fig, ax = plt.subplots()

            ax.barh(
                base_results["Model"],
                base_results["ROC-AUC"]
            )

            ax.set_xlabel("ROC-AUC Score")
            ax.set_title("Model Comparison")

            st.pyplot(fig)


        # -----------------------------
        # Tuned Model Results
        # -----------------------------
        st.subheader("Tuned Model Results")

        st.dataframe(
            output["tuned_results"],
            use_container_width=True
        )


        # -----------------------------
        # Best Model
        # -----------------------------
        st.subheader("Best Model Selected")

        st.success(output["best_model_name"])


        # -----------------------------
        # Test Data
        # -----------------------------
        best_model = output["best_model"]

        X_test = output["X_test"]
        y_test = output["y_test"]


        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        st.subheader("Confusion Matrix")

        y_pred = best_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)


        # -----------------------------
        # ROC Curve (Binary Only)
        # -----------------------------
        if len(np.unique(y_test)) == 2:

            st.subheader("ROC Curve")

            try:

                y_prob = best_model.predict_proba(X_test)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_prob)

                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()

                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

                ax.plot([0,1],[0,1],'--')

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")

                ax.legend()

                st.pyplot(fig)

            except:
                st.warning("ROC curve unavailable for this model.")


        # -----------------------------
        # Feature Importance
        # -----------------------------
        st.subheader("Feature Importance (Top 15)")

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
            ).head(15)

            fig, ax = plt.subplots()

            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )

            ax.invert_yaxis()

            ax.set_title("Top Important Features")

            st.pyplot(fig)

        except:
            st.warning("Feature importance not available for this model.")