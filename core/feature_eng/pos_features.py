from __future__ import annotations

import numpy as np
import pandas as pd

from core.feature_eng.app_features import one_hot_encoder

DATA_PATH = "dataset/"


def pos_cash(feature):
    pos = pd.read_csv(DATA_PATH + 'pos_balance.csv')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    pos["MONTHS_BALANCE"] = abs(pos["MONTHS_BALANCE"])
    cut_points = [0, 2, 4, 12, 24, 36]
    cut_points = cut_points + [pos["MONTHS_BALANCE"].max()]
    labels = ["2MON", "4MON", "12MON", "24MON", "36MON", "ABOVE"]
    pos["MON_INTERVAL"] = pd.cut(pos["MONTHS_BALANCE"], cut_points, labels=labels, include_lowest=True)

    pos["SK_DPD_DIFF"] = pos.SK_DPD - pos.SK_DPD_DEF

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean'],
        'SK_DPD': ['max', 'mean', 'sum', "std"],
        'SK_DPD_DEF': ['max', 'mean', 'sum', "std"],
        "SK_DPD_DIFF": ['max', 'mean', 'sum', "first", "last"],
        'CNT_INSTALMENT': ['std', 'mean', "max", "min"],
        'CNT_INSTALMENT_FUTURE': ['std', 'mean', "max", "min"],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pd.pivot_table(pos, index=["SK_ID_CURR"], columns=["MON_INTERVAL"], aggfunc=aggregations).astype(
        'float32')
    pos_agg.columns = ["POS_" + "_".join(f_).upper() for f_ in pos_agg.columns]
    feature = pd.merge(feature, pos_agg.reset_index(), on="SK_ID_CURR", how="left")

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    pos_agg.columns = pd.Index(['POS_TOTAL' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    feature = pd.merge(feature, pos_agg.reset_index(), on="SK_ID_CURR", how="left")
    return feature
