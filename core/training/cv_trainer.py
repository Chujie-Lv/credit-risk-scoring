from __future__ import annotations

from typing import Any, Dict
from datetime import datetime
import gc
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM does not support special JSON characters in feature names."""
    df.columns = [re.sub(r'[{}[\]",:]+', '_', c) for c in df.columns]
    return df

from core.config import params, seed


def get_default_params(index: int = 0) -> Dict[str, Any]:
    return params[index]


def train_lgbm_cv(
    data: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    *,
    params_idx: int = 0,
    n_splits: int = 5,
    random_state: int = seed,
) -> Dict[str, Any]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 清理列名中的特殊字符
    data = _sanitize_columns(data.copy())
    test = _sanitize_columns(test.copy())

    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()

    feats = [f for f in data.columns if f not in ["SK_ID_CURR", "TARGET"]]

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data, y)):
        X_train, y_train = data[feats].iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = data[feats].iloc[val_idx], y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        clf = lgb.train(
            params[params_idx],
            lgb_train,
            num_boost_round=20000,
            valid_sets=[lgb_valid],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=100),
            ],
        )

        oof_preds[val_idx] = clf.predict(X_valid, num_iteration=clf.best_iteration)

        sub = (
            pd.Series(clf.predict(test[feats], num_iteration=clf.best_iteration))
            .rank(pct=True)
            .values
        )
        sub_preds += sub / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = clf.feature_name()
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print("Fold %2d AUC : %.6f" % (n_fold + 1, roc_auc_score(y_valid, oof_preds[val_idx])))
        del X_train, y_train, X_valid, y_valid
        gc.collect()

    full_auc = roc_auc_score(y, oof_preds)
    print("Full AUC score %.6f" % full_auc)

    test["TARGET"] = sub_preds
    df_oof_preds = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": y, "PREDICTION": oof_preds})[
        ["SK_ID_CURR", "TARGET", "PREDICTION"]
    ]

    now = datetime.now()
    score = str(round(full_auc, 6)).replace(".", "")
    sub_file = (
        "output/submission_lgbm_"
        + score
        + "_"
        + str(now.strftime("%Y-%m-%d-%H-%M"))
        + "_seed_"
        + str(seed)
        + ".csv"
    )
    test[["SK_ID_CURR", "TARGET"]].to_csv(sub_file, index=False)

    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data, y)]

    return {
        "oof_preds": oof_preds,
        "sub_preds": sub_preds,
        "feature_importance_df": feature_importance_df,
        "folds_idx": folds_idx,
        "sub_file": sub_file,
        "df_oof_preds": df_oof_preds,
        "full_auc": full_auc,
    }


__all__ = ["get_default_params", "train_lgbm_cv", "seed"]
