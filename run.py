from src.config import Paths, Settings
from src.data_prep import load_office_actions, load_rejections, load_citations
from src.features import merge_rejections, merge_citations, build_application_table
from src.split_preprocess import make_xy, split_train_test
from src.baseline import compare_models
from src.tuning import tune_xgboost
from src.evaluate import evaluate_model, save_metrics, plot_confusion_matrix, plot_roc_curve



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

    from src.split_preprocess import make_preprocessors
    preprocess_with_scale, preprocess_with_pass = make_preprocessors(cfg.cat_cols, num_cols)

    # 1) Baselines
    baseline_df = compare_models(
        X_train, y_train,
        preprocess_with_scale=preprocess_with_scale,
        preprocess_with_pass=preprocess_with_pass,
        subsample_n=cfg.baseline_subsample_n,
        random_state=cfg.random_state,
    )
    print("\nBaseline comparison (top 5):")
    print(baseline_df.head())

    baseline_df.to_csv(paths.metrics_dir / "baseline_results.csv", index=False)

    # 2) Hyperparameter tuning
    best_model, best_params, best_cv_auc, tuning_df = tune_xgboost(
        X_train, y_train,
        preprocess_with_pass=preprocess_with_pass,
    )

    print("\nBest hyperparameters from tuning:", best_params)
    print("Best CV ROC-AUC from tuning:", best_cv_auc)

    # Save tuning results
    tuning_df.sort_values("mean_test_roc_auc", ascending=False).head(50).to_csv(
        paths.metrics_dir / "tuning_results.csv", index=False
    )

    # 3) Evaluation on test set
    metrics, y_pred, y_proba = evaluate_model(best_model, X_test, y_test)
    print("\nFinal test metrics:", metrics)

    save_metrics(metrics, paths.metrics_dir / "final_metrics.json")
    plot_confusion_matrix(y_test, y_pred, paths.figures_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, paths.figures_dir / "roc_curve.png")

if __name__ == "__main__":
    main()