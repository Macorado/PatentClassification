import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def stratified_subsample(X, y, n, random_state=42):
    """Return a stratified subsample (X_sub, y_sub)"""
    if n is None or n >= len(X):
        return X, y
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y.iloc[idx]

def compare_models(
        X_train,
        y_train,
        preprocess_with_scale,
        preprocess_with_pass,
        subsample_n=200_000,
        random_state=42,
        cv=5,
        n_jobs=-1,
):
    """Compare baseline models using CV ROC-AUC. Returns results_df sorted by mean AUC."""
    X_sub, y_sub= stratified_subsample(X_train, y_train, subsample_n, random_state)

    models = {
        "LogReg": Pipeline(steps=[
            ("preprocess", preprocess_with_scale),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "RandomForest": Pipeline(steps=[
            ("preprocess", preprocess_with_pass),
            ("model", RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)),
        ]),
        "GradBoost": Pipeline(steps=[
            ("preprocess", preprocess_with_pass),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]),
        "XGBoost": Pipeline(steps=[
            ("preprocess", preprocess_with_pass),
            ("model", XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=n_jobs)),
        ]),
        "LightGBM": Pipeline(steps=[
            ("preprocess", preprocess_with_pass),
            ("model", LGBMClassifier(random_state=random_state, n_jobs=n_jobs)),
        ]),
    }

    results = []
    for name, pipe in models.items():
        start = time.time()
        scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        elapsed = time.time() - start

        results.append({
            "model": name,
            "cv_auc_mean": float(np.mean(scores)),
            "cv_auc_std": float(np.std(scores)),
            "seconds": elapsed,
            "minutes": elapsed / 60,
            "n_rows": int(len(X_sub)),
        })

    results_df = pd.DataFrame(results).sort_values(by="cv_auc_mean", ascending=False).reset_index(drop=True)
    return results_df