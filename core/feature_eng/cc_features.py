from __future__ import annotations

import numpy as np
import pandas as pd

from core.feature_eng.app_features import one_hot_encoder

DATA_PATH = "dataset/"


def credit_card_balance(feature):
    cc = pd.read_csv(DATA_PATH + 'cc_balance.csv')
    # Preprocessing
    cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
    cc.loc[cc['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=False)
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    cc["MONTHS_BALANCE"] = abs(cc["MONTHS_BALANCE"])
    cut_points = [0, 2, 4, 12, 24, 36]
    cut_points = cut_points + [cc["MONTHS_BALANCE"].max()]
    labels = ["2MON", "4MON", "12MON", "24MON", '36MON', "ABOVE"]
    cc["MON_INTERVAL"] = pd.cut(cc["MONTHS_BALANCE"], cut_points, labels=labels, include_lowest=True)

    cc["SK_DPD_DIFF"] = cc.SK_DPD - cc.SK_DPD_DEF
    cc["AMT_PAYMENT_MIN_REGULARITY_PERC"] = cc["AMT_INST_MIN_REGULARITY"] / cc["AMT_PAYMENT_CURRENT"]
    cc["DRAWINGS_ATM_PERC"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / cc["CNT_DRAWINGS_ATM_CURRENT"]
    cc["DRAWING_PERC"] = cc["AMT_DRAWINGS_CURRENT"] / cc["CNT_DRAWINGS_CURRENT"]
    cc["DRAWINGS_OTHER_PERC"] = cc["AMT_DRAWINGS_OTHER_CURRENT"] / cc["CNT_DRAWINGS_OTHER_CURRENT"]
    cc["DRAWINGS_POS_PERC"] = cc["AMT_DRAWINGS_POS_CURRENT"] / cc["CNT_DRAWINGS_POS_CURRENT"]
    cc["PAYMENT_INSTALMENT_PERC"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / cc["CNT_INSTALMENT_MATURE_CUM"]

    cc['AMT_DRAWINGS_ATM_PER'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_DRAWINGS_CURRENT']
    cc['CNT_DRAWINGS_ATM_PER'] = cc['CNT_DRAWINGS_ATM_CURRENT'] / cc['CNT_DRAWINGS_CURRENT']
    cc['AMT_DRAWINGS_POS_CURRENT_PER'] = cc['AMT_DRAWINGS_POS_CURRENT'] / cc['CNT_DRAWINGS_ATM_CURRENT']
    cc['CNT_DRAWINGS_POS_AVG'] = cc['AMT_DRAWINGS_POS_CURRENT'] / cc['CNT_DRAWINGS_POS_CURRENT']
    cc['AMT_RECIVABLE_BALANCE'] = cc['AMT_RECIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['AMT_RECIVABLE_PER'] = cc['AMT_RECIVABLE'] / cc['AMT_TOTAL_RECEIVABLE']

    aggregations = {
        'MONTHS_BALANCE': ["min", 'max'],
        'AMT_BALANCE': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum', "min"],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum', "std"],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum', "std"],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum', "std"],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'sum', "std"],
        'AMT_PAYMENT_CURRENT': ['std', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean', 'sum', "std"],
        'AMT_RECEIVABLE_PRINCIPAL': ["std", 'max', 'mean', 'sum'],
        'AMT_RECIVABLE': ["std", 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ["std", 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum', "std"],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum', "std"],
        'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum', "std"],
        'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean', 'sum', "std"],
        'SK_DPD': ['max', 'mean', 'sum', "std"],
        'SK_DPD_DEF': ['max', 'mean', 'sum', "std"],
        "SK_DPD_DIFF": ['first', 'max', 'mean', 'sum'],
        "AMT_PAYMENT_MIN_REGULARITY_PERC": ["min", 'max', 'mean', 'sum'],
        "DRAWING_PERC": ["min", 'max', 'mean', 'sum'],
        "DRAWINGS_ATM_PERC": ['max', 'mean', 'sum', "std"],
        "DRAWINGS_OTHER_PERC": ['max', 'mean', 'sum', "std"],
        "DRAWINGS_POS_PERC": ['max', 'mean', 'sum', "std"],
        "PAYMENT_INSTALMENT_PERC": ['max', 'mean', 'sum', "std"],
        'AMT_DRAWINGS_ATM_PER': ['max', 'mean', 'std'],
        'CNT_DRAWINGS_ATM_PER': ['max', 'mean', 'std'],
        'AMT_DRAWINGS_POS_CURRENT_PER': ['max', 'mean', 'std'],
        'CNT_DRAWINGS_POS_AVG': ['max', 'mean', 'std'],
        'AMT_RECIVABLE_BALANCE': ['max', 'mean', 'std'],
        'AMT_RECIVABLE_PER': ['max', 'mean', 'std'],
    }
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    cc_agg = pd.pivot_table(cc, index=["SK_ID_CURR"], columns=["MON_INTERVAL"], aggfunc=aggregations).astype('float32')
    cc_agg.columns = ["CC_PIVOT_" + "_".join(f_).upper() for f_ in cc_agg.columns]
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    feature = pd.merge(feature, cc_agg.reset_index(), on="SK_ID_CURR", how="left")

    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    cc_agg.columns = pd.Index(['CC_TOTAL_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    feature = pd.merge(feature, cc_agg.reset_index(), on="SK_ID_CURR", how="left")
    return feature
