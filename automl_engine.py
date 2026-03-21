import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier


# ------------------------------
# Clean Data (FIXED)
# ------------------------------
def auto_clean_data(df, target_column):

    df = df.copy()

    # Drop ID-like
    df = df.loc[:, df.nunique() / len(df) < 0.95]

    # Drop high missing
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Drop high cardinality
    df = df.loc[:, df.nunique() < 100]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y, cat_cols, num_cols


# ------------------------------
# Preprocessor
# ------------------------------
def build_preprocessor(cat_cols, num_cols):

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


# ------------------------------
# Train Models
# ------------------------------
def train_models(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "DT": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "GB": GradientBoostingClassifier(),
        "XGB": XGBClassifier(eval_metric="logloss")
    }

    results = []

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        try:
            y_prob = pipe.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob[:, 1])
        except:
            roc = np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc
        })

    return pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False), X_test, y_test


# ------------------------------
# Main Pipeline
# ------------------------------
def run_full_pipeline(df, target_column):

    if len(df) > 50000:
        df = df.sample(50000, random_state=42)

    X, y, cat_cols, num_cols = auto_clean_data(df, target_column)

    preprocessor = build_preprocessor(cat_cols, num_cols)

    results, X_test, y_test = train_models(X, y, preprocessor)

    best_model_name = results.iloc[0]["Model"]

    model_map = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "DT": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "GB": GradientBoostingClassifier(),
        "XGB": XGBClassifier(eval_metric="logloss")
    }

    final_model = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model_map[best_model_name])
    ])

    final_model.fit(X, y)

    return {
        "base_results": results,
        "tuned_results": results,
        "best_model": final_model,
        "X_test": X_test,
        "y_test": y_test,
        "problem_type": "classification"
    }