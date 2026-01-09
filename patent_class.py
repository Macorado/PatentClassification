# %%

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, roc_auc_score, roc_curve, f1_score, recall_score, precision_score,auc, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from pprint import pprint
import time

# %%


#Office Actions
df_oa = pd.read_pickle('data\office_actions.pkl')

# Convert Dates
df_oa['mail_dt'] = pd.to_datetime(df_oa['mail_dt'])

# Drop columns early to save RAM
drop_cols = ['header_missing', 'fp_missing', 'closing_missing', 'signature_type']
df_oa.drop(columns=[c for c in drop_cols if c in df_oa.columns], inplace=True)

df_oa.head(10)

# %%
df_oa.shape

# %%
df_oa.dtypes

# %%
#Rejections
df_rj = pd.read_pickle('data/rejections.pkl')


df_rj.head(10)

# %%
df_rj.dtypes

# %%
df_rj.shape

# %%
import gc
# Process rejections

# Convert 'action_type' strings to binary
df_rj['is_102'] = (df_rj['action_type'].astype(str) == '102').astype(int)
df_rj['is_103'] = (df_rj['action_type'].astype(str) == '103').astype(int)

# Aggregate
# Compress rejection rows into summary stats per ifw_number
rj_agg = df_rj.groupby('ifw_number').agg({
    'action_type': 'count',     # Total issues raised
    'alice_in' : 'max',         # Any Alice rejection
    'bilski_in': 'max',         # Any Bilski rejection
    'is_102': 'sum',            # Count of 102 rejections
    'is_103': 'sum'             # Count of 103 rejections
}).rename(columns={'action_type': 'rejection_count'})

# merge summaries onto the main dataframe
df_oa = df_oa.merge(rj_agg, on='ifw_number', how='left')

# Verify by printing first 5 rows with rejections
# ifw_number should match Office Action IDs
# rejection_count should be a number not NaN
# is_103 should show count of how many times the examiner cited "obviousness" in that letter
print(df_oa[df_oa['rejection_count'] > 0][['ifw_number', 'mail_dt', 'rejection_count', 'is_103', 'alice_in']].head())

# Fill NaNs (office action had 0 rejections)
fill_cols = ['rejection_count', 'alice_in', 'bilski_in', 'is_102', 'is_103']
df_oa[fill_cols] = df_oa[fill_cols].fillna(0)




# %%
# Deleting rejections dataframe after merge
del df_rj
del rj_agg
gc.collect()

# %%
#Citations
df_ct = pd.read_pickle('data/citations.pkl')

# %%
df_ct.head(10)

# %%
df_ct.shape

# %%
df_ct.dtypes

# %%
# Aggregate: Count citations per Application ID
ct_agg = df_ct.groupby('app_id').agg({
    'app_id': 'count'
}).rename(columns={'app_id': 'citation_count'})

# Merge onto main dataframe
df_oa = df_oa.merge(ct_agg, on='app_id', how='left')
df_oa['citation_count'] = df_oa['citation_count'].fillna(0)

# %%
# Delete citations dataframe
del df_ct
del ct_agg
gc.collect()

# %%
df_oa.head(5)

# %%
pd.set_option('display.max_columns', None)

# %%
# Aggregate signals to application level
app_agg = (
    df_oa.groupby("app_id", as_index=False)
    .agg(
        app_id_count=("app_id", "size"),

        rejection_fp_mismatch=("rejection_fp_mismatch", "max"),
        rejection_101=("rejection_101", "max"),
        rejection_102=("rejection_102", "max"),
        rejection_103=("rejection_103", "max"),
        rejection_112=("rejection_112", "max"),
        rejection_dp=("rejection_dp", "max"),
        objection=("objection", "max"),

        # Target:
        allowed_claims=("allowed_claims", "max"),

        cite102_gt1=("cite102_gt1", "max"),
        cite103_gt3=("cite103_gt3", "max"),
        cite103_eq1=("cite103_eq1", "max"),
        alice_in=("alice_in", "max"),
        bilski_in=("bilski_in", "max"),

        cite103_max=("cite103_max", "sum"),
        rejection_count=("rejection_count", "sum"),
        is_102=("is_102", "sum"),
        is_103=("is_103", "sum"),

        first_mail_date=("mail_dt", "min"),
        last_mail_date=("mail_dt", "max"),

        # carry forward citation_count
        citation_count=("citation_count", "max"),
    )
)

# Document type counts (CTNF/CTFR) per app_id
doc_counts = (
    df_oa.groupby(["app_id", "document_cd"]).size().unstack(fill_value=0).reset_index()
)

# Rename doc columns to *_count
doc_counts = doc_counts.rename(columns={c: f"{c}_count" for c in doc_counts.columns if c != "app_id"})

# Status "identity" columns (taking first observed with app_id)
# These should not change across ifw_number hence they are used for same app_id
static_cols = ["app_id", "art_unit", "uspc_class", "uspc_subclass"]
base_static = (
    df_oa.sort_values(["app_id", "mail_dt"]).drop_duplicates(subset=["app_id"], keep="first")[static_cols]
)

# Creation of final dataset
final_oa = (
    base_static.merge(app_agg, on="app_id", how="left").merge(doc_counts, on="app_id", how="left")
)

# Prosecution length feature:
final_oa["prosecution_days"] = (final_oa["last_mail_date"] - final_oa["first_mail_date"]).dt.days

final_oa.head()

# %%
# --- cleanup ---
del app_agg, doc_counts, base_static
gc.collect()

# %%
# Creating target and feature matrix
target = "allowed_claims"

drop_cols = ["app_id", "first_mail_date", "last_mail_date"]  # not modeling on raw IDs and dates
X = final_oa.drop(columns=[target] + drop_cols).copy()
y = final_oa[target].astype(int).copy()

# Categorical columns
cat_cols = ["art_unit", "uspc_class", "uspc_subclass"]
for c in cat_cols:
  X[c] = X[c].astype(str)

num_cols = [c for c in X.columns if c not in cat_cols]

# Single split (test set stays untouched until final evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
print("Train positive rate:", y_train.mean(), "| Test positive rate:", y_test.mean())


# %%
# Preprocessing
# One hot encode cats, scale numeric for when necessary

cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess_with_scale = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop"
)

preprocess_with_pass = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

# %%
# Stratified subsample for baseline model comparison
SUB_N = 200_000

sss = StratifiedShuffleSplit(n_splits=1, train_size=SUB_N, random_state=42)
sub_idx, _ = next(sss.split(X_train, y_train))

X_train_sub = X_train.iloc[sub_idx]
y_train_sub = y_train.iloc[sub_idx]

print("Subsample shape", X_train_sub.shape)
print("Class balance (sub):", y_train_sub.mean(), " | (full train):", y_train.mean())

# %%
# Baseline model comparison using CV on subsample
# Compare mean ROC-AUC across all models and track runtime

models = {
    "LogReg" : Pipeline(steps=[
        ("preprocess", preprocess_with_scale),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "RandomForest" : Pipeline(steps=[
        ("preprocess", preprocess_with_pass),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ]),
    "GradBoost" : Pipeline(steps=[
        ("preprocess", preprocess_with_pass),
        ("model", GradientBoostingClassifier(random_state=42))
    ]),
    "XGBoost" : Pipeline(steps=[
        ("preprocess", preprocess_with_pass),
        ("model", XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ))
    ]),
    "LightGBM" : Pipeline(steps=[
        ("preprocess", preprocess_with_pass),
        ("model", lgb.LGBMClassifier(random_state=42))
    ]),
}

results = []

for name, pipe in models.items():
  start = time.time()

  scores = cross_val_score(
      pipe,
      X_train_sub, y_train_sub,
      cv=5,
      scoring="roc_auc",
      n_jobs=-1
  )

  elapsed = time.time() - start

  results.append({
      "model" : name,
      "cv_auc_mean" : float(np.mean(scores)),
      "cv_auc_std" : float(np.std(scores)),
      "seconds" : elapsed,
      "minutes" : elapsed / 60.0
  })

results_df = pd.DataFrame(results).sort_values("cv_auc_mean", ascending=False)
results_df

# %%
best_row = results_df.iloc[0]
print(
    f"Best baseline model: {best_row['model']} | "
    f"AUC={best_row['cv_auc_mean']:.3f} +/- {best_row['cv_auc_std']:.3f} | "
    f"Time={best_row['seconds']:.2f} sec"
)

# %%
# --- cleanup ---
del X_train_sub, y_train_sub, results, results_df, models
gc.collect()

# %%
#Establish weights
neg, pos = np.bincount(y_train)
scale_pos_weight = neg/pos

# %%
#XGBoost was best performing in initial tests. Create pipeline and hyper tune parameters
pipe = Pipeline([

    ('preproc', preprocess_with_pass),
    ('mod', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state = 115,
        n_jobs = -1
        ))
    ])

param_grid = [
    {
    'mod__n_estimators' : [100,200,300],
    'mod__max_depth': [3,5],
    'mod__gamma': [0,0.5,1],
    'mod__subsample' : [0.7,0.9,1.0],
    'mod__learning_rate' : [0.05,0.1]
    }
]

scoring = {
    'roc_auc' : 'roc_auc',
    'accuracy' : 'accuracy',
    'pr_auc' : 'average_precision',
    'recall' : 'recall',
    'f1' : 'f1'
}

rscv = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    cv = 5,
    n_iter = 30,
    scoring = scoring,
    refit='roc_auc',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
    random_state=115
)


#Fit RSCV to training data
rscv.fit(X_train,y_train)


#Store results dataframe
df_results = pd.DataFrame(rscv.cv_results_)

#Output results
print(f"Optimal Parameters:{rscv.best_params_}")
print(f"Mean Training Fit Time for Optimal Model:{rscv.cv_results_['mean_fit_time'][rscv.best_index_]}")
print(f"Training CV ROC/AUC:{rscv.best_score_}")
print(f"Test ROC/AUC:{rscv.score(X_test,y_test)}")

# %%
#Specify metrics and sort result df
metric_cols = [
    'mean_test_roc_auc',
    'mean_test_pr_auc',
    'mean_test_accuracy',
    'mean_test_recall',
    'mean_test_f1',
    'mean_fit_time',
    'params'
]

df_results = (
    df_results[metric_cols]
    .sort_values(by='mean_test_roc_auc', ascending=False)
    .reset_index(drop=True)
)

# %%
df_results.head()

# %%
rscv.best_params_

# %%
#Fit hypertuned model on entire test set no cv and evaluate
best_model = rscv.best_estimator_

y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)[:,1]


results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Model": "XGBoost"
}

final_results_df = pd.DataFrame([results])

final_results_df.head()

# %%
#Plot confusion matrix
cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=best_model.classes_)
disp.plot(cmap='Greens')

# %%
#Generate ROC curve and plot
fpr,tpr, thresholds = roc_curve(y_test,y_proba)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0,1],[0,1], 'k--', label = 'No Skill')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve XGBoost')
plt.legend()
plt.show()

# %%
#Data in wide format need to melt into long for visualization
final_results_df_long = pd.melt(
    final_results_df,
    id_vars='Model',
    value_vars=['Accuracy','ROC AUC', 'Precision','Recall','F1'],
    var_name='Metric',
    value_name='Score'
)

# %%
final_results_df_long.head()

# %%
#Produce radar plot comparing all metrics associated to final model.
fig = px.line_polar(
    final_results_df_long,
    r='Score',
    theta='Metric',
    line_close=True,
    template='plotly_dark'
    )


fig.update_layout(
    title='XGBoost Metrics',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0,1]
        ),
        angularaxis=dict(
            direction='clockwise'
        )
    ),
)
fig.update_traces(fill='toself')
fig.show()


