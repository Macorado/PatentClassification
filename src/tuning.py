import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def compute_scale_pos_weight(y):
    neg, pos = np.bincount(y)
    return neg / pos

def tune_xgboost(
        X_train,
        y_train,
        preprocess_with_pass,
        random_state=42,
        n_iter=30,
        cv=5,
        n_jobs=-1
):
    scale_pos_weight = compute_scale_pos_weight(y_train)

    pipe = Pipeline([
        ("preproc", preprocess_with_pass),
        ("mod", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        ))
    ])

    param_distributions = {
        "mod__n_estimators": [100, 200, 300, 400, 500],
        "mod__max_depth": [3, 5, 7, 9, 11],
        "mod__gamma": [0, 0.1, 0.25, 0.5, 0.75, 1],
        "mod__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "mod__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    }

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "pr_auc": "average_precision",
        "recall": "recall",
        "f1": "f1",
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit="roc_auc",
        return_train_score=True,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    search.fit(X_train, y_train)

    results_df = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, float(search.best_score_), results_df