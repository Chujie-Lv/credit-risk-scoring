from __future__ import annotations

from typing import Tuple

import pandas as pd

from core.feature_eng.app_features import application_train_test
from core.feature_eng.bureau_features import bureau_and_balance
from core.feature_eng.prev_app_features import previous_applications
from core.feature_eng.pos_features import pos_cash
from core.feature_eng.inst_features import installments_payments
from core.feature_eng.cc_features import credit_card_balance
from core.feature_eng.cross_ratios import association_perc
from core.feature_eng.recency_features import top_k_ins_by_prev
from core.feature_eng.col_filter import drop_col


def build_dataset(num_rows: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build the full feature matrix for LightGBM training.

    Returns:
        data : Training feature DataFrame (TARGET non-null rows, columns pruned)
        test : Test feature DataFrame
        y    : Training labels
        ids  : Training SK_ID_CURR
    """
    df = application_train_test(num_rows)
    df = bureau_and_balance(df)
    df = previous_applications(df)
    df = pos_cash(df)
    df = installments_payments(df)
    df = credit_card_balance(df)
    df = association_perc(df)
    df = top_k_ins_by_prev(df)

    data = df[~df.TARGET.isnull()]
    test = df[df.TARGET.isnull()]
    del df

    ids = data["SK_ID_CURR"]
    y = data["TARGET"]

    valid_drop_cols = [f for f in drop_col if f in data.columns]
    test = test.drop(valid_drop_cols, axis=1)
    data = data.drop(valid_drop_cols, axis=1)

    return data, test, y, ids
