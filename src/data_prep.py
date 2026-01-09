import pandas as pd

def load_office_actions(path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["mail_dt"] = pd.to_datetime(df["mail_dt"])

    drop_cols = ["header_missing", "fp_missing", "closing_missing", "signature_type"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

def load_rejections(path) -> pd.DataFrame:
    return pd.read_pickle(path)

def load_citations(path) -> pd.DataFrame:
    return pd.read_pickle(path)