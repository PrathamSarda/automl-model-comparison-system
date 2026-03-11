# AutoML Model Comparison System

An automated machine learning system that allows users to upload a dataset and automatically train multiple ML models, perform hyperparameter tuning, and compare model performance.

## Features

- Automatic dataset cleaning
- Automatic preprocessing pipeline
- Model comparison across multiple ML algorithms
- Hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy, F1-score, and ROC-AUC
- Visualizations including:
  - Model comparison chart
  - Confusion matrix
  - ROC curve
  - Feature importance

## Models Used

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

## Technologies

- Python
- Scikit-learn
- XGBoost
- Streamlit
- Pandas
- Matplotlib
- Seaborn

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt