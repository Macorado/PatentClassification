from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_xy(final_df, target_col, drop_cols, cat_cols):
    X = final_df.drop(columns=[target_col, *drop_cols]).copy()
    y = final_df[target_col].astype(int).copy()

    for c in cat_cols:
        X[c] = X[c].astype(str)
    
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols

def make_preprocessors(cat_cols, num_cols):
    cat = OneHotEncoder(handle_unknown="ignore")

    with_scale = ColumnTransformer([
        ("cat", cat, list(cat_cols)),
        ("num", StandardScaler(), list(num_cols)),
    ])

    passthrough = ColumnTransformer([
        ("cat", cat, list(cat_cols)),
        ("num", "passthrough", list(num_cols)),
    ])

    return with_scale, passthrough

def split_train_test(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)