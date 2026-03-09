from core.feature_eng.builder import build_dataset
from core.training.cv_trainer import train_lgbm_cv
from core.viz.charts import display_importances, display_roc_curve


def main() -> None:
    data, test, y, ids = build_dataset()

    results = train_lgbm_cv(data=data, test=test, y=y, ids=ids)
    print(f"Submission saved to {results['sub_file']}")

    display_importances(results["feature_importance_df"])
    display_roc_curve(y_=y, oof_preds_=results["oof_preds"], folds_idx_=results["folds_idx"])


if __name__ == "__main__":
    main()
