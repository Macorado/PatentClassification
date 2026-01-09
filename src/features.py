import pandas as pd

def merge_rejections(df_oa: pd.DataFrame, df_rj: pd.DataFrame) -> pd.DataFrame:
    df_rj = df_rj.copy()
    df_rj["is_102"] = (df_rj["action_type"].astype(str) == "102").astype(int)
    df_rj["is_103"] = (df_rj["action_type"].astype(str) == "103").astype(int)

    rj_agg = (
        df_rj.groupby("ifw_number")
        .agg(
            rejection_count=("action_type", "count"),
            alice_in=("alice_in", "max"),
            bilski_in=("bilski_in", "max"),
            is_102=("is_102", "max"),
            is_103=("is_103", "max"),
        )
    )

    out = df_oa.merge(rj_agg, on="ifw_number", how="left")

    fill_cols = ["rejection_count", "alice_in", "bilski_in", "is_102", "is_103"]
    out[fill_cols] = out[fill_cols].fillna(0)
    return out

def merge_citations(df_oa: pd.DataFrame, df_ct: pd.DataFrame) -> pd.DataFrame:
    ct_agg = df_ct.groupby("app_id").size().rename("citation_count").reset_index()
    out = df_oa.merge(ct_agg, on="app_id", how="left")
    out["citation_count"] = out["citation_count"].fillna(0)
    return out

def build_application_table(df_oa: pd.DataFrame) -> pd.DataFrame:
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

            citation_count=("citation_count", "max"),
        )
    )

    # Doc type counts
    doc_counts = (
        df_oa.groupby(["app_id", "document_cd"]).size().unstack(fill_value=0).reset_index()
    )
    doc_counts = doc_counts.rename(
        columns={c: f"{c}_count" for c in doc_counts.columns if c != "app_id"}
    )

    static_cols = ["app_id", "art_unit", "uspc_class", "uspc_subclass"]
    base_static = (
        df_oa.sort_values(["app_id", "mail_dt"])
        .drop_duplicates(subset=["app_id"], keep="first")[static_cols]
    )

    final_oa = (
        base_static
        .merge(app_agg, on="app_id", how="left")
        .merge(doc_counts, on="app_id", how="left")
    )

    final_oa["prosecution_days"] = (final_oa["last_mail_date"] - final_oa["first_mail_date"]).dt.days
    return final_oa
