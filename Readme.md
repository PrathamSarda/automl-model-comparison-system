AutoML Model Comparison System

## 🌐 Live Demo
https://automl-model-comparison-system-rxcvjbtwa2vu69pfmgtzfe.streamlit.app/

An interactive AutoML web application that allows users to upload datasets, automatically preprocess data, train multiple machine learning models, perform hyperparameter tuning, and compare model performance.

🚀 Features
Automatic dataset cleaning (missing values, ID removal)
Automated preprocessing pipeline:
Imputation
Scaling
One-hot encoding
Trains multiple ML models automatically
Hyperparameter tuning using RandomizedSearchCV
Supports:
Binary classification
Multiclass classification
Interactive visualizations:
Model comparison chart
Confusion matrix
ROC curve (for binary classification)
Feature importance
Prediction interface with:
Top important features
Risk scoring (Low / Medium / High)


🤖 Models Used
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Gradient Boosting
XGBoost


🛠️ Technologies
Python
Scikit-learn
XGBoost
Streamlit
Pandas
NumPy
Matplotlib
Seaborn

📊 Workflow

Upload Dataset
↓
EDA Analysis
↓
Data Cleaning & Preprocessing
↓
Train 6 ML Models
↓
Select Top 3 Models
↓
Hyperparameter Tuning
↓
Model Comparison
↓
Best Model Selection
↓
Prediction & Risk Scoring


▶️ How to Run Locally

git clone https://github.com/PrathamSarda/automl-model-comparision-system.git
cd automl-model-comparision-system

pip install -r requirements.txt
streamlit run app.py
