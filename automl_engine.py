import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier


# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
def auto_clean_data(df, target_column):

    df = df.copy()

    threshold = 0.95
    n_rows = len(df)

    id_like_cols = [
        col for col in df.columns
        if df[col].nunique() / n_rows > threshold
    ]

    id_like_cols = [col for col in id_like_cols if col != target_column]

    df.drop(columns=id_like_cols, inplace=True)

    missing_percent = df.isnull().mean()
    high_missing_cols = missing_percent[missing_percent > 0.5].index.tolist()

    df.drop(columns=high_missing_cols, inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    return X, y, categorical_cols, numerical_cols


# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
def build_preprocessor(categorical_cols, numerical_cols):

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor


# --------------------------------------------------
# Base Models
# --------------------------------------------------
def train_base_models(X, y, preprocessor):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),

        "KNN": KNeighborsClassifier(),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # probability safe
        try:
            y_prob = pipeline.predict_proba(X_test)
        except:
            y_prob = None

        # ROC-AUC safe
        try:
            if y_prob is not None:

                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:,1])

                else:
                    roc_auc = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr"
                    )
            else:
                roc_auc = np.nan

        except:
            roc_auc = np.nan


        results.append({

            "Model": name,

            "Accuracy": accuracy_score(y_test, y_pred),

            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "F1": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "ROC-AUC": roc_auc
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(by="ROC-AUC", ascending=False)


# --------------------------------------------------
# Model Configurations
# --------------------------------------------------
MODEL_CONFIGS = {

    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "params": {"model__C": [0.1, 1, 10]}
    },

    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"model__n_neighbors": [3,5,7]}
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "params": {
            "model__max_depth":[None,10,20],
            "model__min_samples_split":[2,5]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params":{
            "model__n_estimators":[100,200],
            "model__max_depth":[None,10]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params":{
            "model__n_estimators":[100,200],
            "model__learning_rate":[0.05,0.1]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "params":{
            "model__n_estimators":[100,200],
            "model__learning_rate":[0.05,0.1],
            "model__max_depth":[3,5]
        }
    }
}


# --------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------
def tune_top_models(X, y, preprocessor, base_results):

    top_models = base_results["Model"].head(3).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tuned_results = []

    for model_name in top_models:

        config = MODEL_CONFIGS[model_name]

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", config["model"])
        ])

        grid_search = GridSearchCV(
            pipeline,
            config["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        tuned_results.append({

            "Model": model_name,

            "Best Params": grid_search.best_params_,

            "Accuracy": accuracy_score(y_test, y_pred),

            "F1": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )
        })

    return pd.DataFrame(tuned_results).sort_values(by="Accuracy", ascending=False)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------
def run_full_pipeline(df, target_column):

    X, y, cat_cols, num_cols = auto_clean_data(df, target_column)

    preprocessor = build_preprocessor(cat_cols, num_cols)

    base_results = train_base_models(X, y, preprocessor)

    tuned_results = tune_top_models(
        X,
        y,
        preprocessor,
        base_results
    )

    best_model_name = tuned_results.iloc[0]["Model"]

    best_config = MODEL_CONFIGS[best_model_name]

    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", best_config["model"])
    ])

    final_pipeline.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return {

        "base_results": base_results,

        "tuned_results": tuned_results,

        "best_model": final_pipeline,

        "best_model_name": best_model_name,

        "X_test": X_test,

        "y_test": y_test
    }