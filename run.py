from src.config import Paths, Settings
from src.data_prep import load_office_actions, load_rejections, load_citations
from src.features import merge_rejections, merge_citations, build_application_table
from src.split_preprocess import make_xy, split_train_test


def main():
    paths = Paths()
    cfg = Settings()

    # Ensure output dirs exist
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)

    df_oa = load_office_actions(paths.office_actions)
    df_rj = load_rejections(paths.rejections)
    df_ct = load_citations(paths.citations)

    df_oa = merge_rejections(df_oa, df_rj)
    df_oa = merge_citations(df_oa, df_ct)

    final_oa = build_application_table(df_oa)

    X, y, num_cols = make_xy(final_oa, cfg.target_col, cfg.drop_cols, cfg.cat_cols)
    X_train, X_test, y_train, y_test = split_train_test(X, y, cfg.test_size, cfg.random_state)

    print("final_oa:", final_oa.shape)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Positive rate train:", y_train.mean(), "test:", y_test.mean())

if __name__ == "__main__":
    main()