from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from core.feature_eng.app_features import one_hot_encoder

DATA_PATH = "dataset/"


def get_bureau_balance():
    data = pd.read_csv(DATA_PATH + 'credit_history_monthly.csv')
    data["STATUS"], uniques = pd.factorize(data["STATUS"])
    data["MONTHS_BALANCE"] = abs(data["MONTHS_BALANCE"])
    cut_points = [0, 2, 4, 12, 24, 36]
    cut_points = cut_points + [data["MONTHS_BALANCE"].max()]
    labels = ["2MON", "4MON", "12MON", "24MON", "36MON", "ABOVE"]
    data["MON_INTERVAL"] = pd.cut(data["MONTHS_BALANCE"], cut_points, labels=labels, include_lowest=True)
    feature = pd.pivot_table(data, index=["SK_ID_BUREAU"], columns=["MON_INTERVAL"], values=["STATUS"],
                             aggfunc=[np.max, np.mean, np.std]).astype('float32')
    feature.columns = ["_".join(f_).upper() for f_ in feature.columns]

    bb_agg = data.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['min', 'max', 'size']}).astype('float32')
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    feature = pd.merge(feature, bb_agg, how="left", left_index=True, right_index=True)
    return feature, feature.columns.tolist()


def bureau_and_balance(feature):
    data = pd.read_csv(DATA_PATH + 'credit_history.csv')
    # Preprocessing
    data.loc[data['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    data.loc[data['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
    data.loc[data['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan

    df, bureau_cat = get_bureau_balance()
    data = pd.merge(data, df, left_on="SK_ID_BUREAU", right_index=True, how="left")
    del data["SK_ID_BUREAU"], data["CREDIT_CURRENCY"]

    temp = data.groupby("CREDIT_TYPE")["AMT_ANNUITY"].mean()
    data["CREDIT_TYPE_AMT_ANNUITY"] = data["CREDIT_TYPE"].map(temp)

    data, cat_cols = one_hot_encoder(data)

    data['CREDICT_SUM_LEFT'] = data['AMT_CREDIT_SUM'] - data['AMT_CREDIT_SUM_DEBT'] - data['AMT_CREDIT_SUM_LIMIT']
    data['DAYS_CREDIT_LAST'] = data['DAYS_CREDIT_ENDDATE'] - data['CREDIT_DAY_OVERDUE']
    data['DAYS_CREDIT_EARLY'] = data['DAYS_CREDIT_ENDDATE'] - data['DAYS_ENDDATE_FACT']

    data['CREDICT_SUM_DEBT_PERC'] = data['AMT_CREDIT_SUM_DEBT'] / data['AMT_CREDIT_SUM']
    data['CREDICT_SUM_LIMIT_PERC'] = data['AMT_CREDIT_SUM_LIMIT'] / data['AMT_CREDIT_SUM']
    data['CREDICT_SUM_OVERDUE_PERC'] = data['AMT_CREDIT_SUM_OVERDUE'] / data['AMT_CREDIT_SUM']
    data['ANNUITY_CREDIT_PERC'] = data['AMT_ANNUITY'] / data['DAYS_CREDIT']
    data['ANNUITY_CREDIT_SUM_PERC'] = data['AMT_ANNUITY'] / data['AMT_CREDIT_SUM']

    num_aggregations = {
        'DAYS_CREDIT': ['max', 'mean', 'sum', "std"],
        'DAYS_CREDIT_ENDDATE': ['max', 'mean', 'sum', "std"],
        'DAYS_CREDIT_UPDATE': ['max', 'mean', 'sum', "std"],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum', "std"],
        'AMT_ANNUITY': ["first", "sum", 'max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'CREDICT_SUM_LEFT': ['max', 'mean', 'sum', "std"],
        'DAYS_CREDIT_LAST': ['max', 'mean', 'sum', "std"],
        'DAYS_CREDIT_EARLY': ['max', 'mean', 'sum', "std"],
        'CREDIT_TYPE_AMT_ANNUITY': ['mean', "std"],
        'CREDICT_SUM_DEBT_PERC': ['max', 'mean', 'sum', "std"],
        'CREDICT_SUM_LIMIT_PERC': ['max', 'mean', 'sum', "std"],
        'CREDICT_SUM_OVERDUE_PERC': ['max', 'mean', 'sum', "std"],
        'ANNUITY_CREDIT_PERC': ['max', 'mean', 'sum', "std"],
        'ANNUITY_CREDIT_SUM_PERC': ['max', 'mean', 'sum', "std"],
    }

    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    result = data.groupby('SK_ID_CURR').agg(dict(num_aggregations, **cat_aggregations)).astype('float32')
    result.columns = ["".join(_f) for _f in result.columns]

    active_flag_col = "CREDIT_ACTIVE_Active"
    if active_flag_col in data.columns:
        active = data[data[active_flag_col] == 1]
        active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations).astype("float32")
        active_agg.columns = pd.Index(
            ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
        )
        result = pd.merge(result, active_agg, right_index=True, left_index=True, how="left")

    closed_flag_col = "CREDIT_ACTIVE_Closed"
    if closed_flag_col in data.columns:
        closed = data[data[closed_flag_col] == 1]
        closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations).astype("float32")
        closed_agg.columns = pd.Index(
            ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
        )
        result = pd.merge(result, closed_agg, right_index=True, left_index=True, how="left")

    lastdate = data[data['DAYS_CREDIT'] > -30]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['30DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    result = pd.merge(result, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = data[data['DAYS_CREDIT'] > -90]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['90DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    result = pd.merge(result, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = data[data['DAYS_CREDIT'] > -120]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['120DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    result = pd.merge(result, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = data[data['DAYS_CREDIT'] > -365]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['365DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    result = pd.merge(result, lastdate_agg, right_index=True, left_index=True, how="left")

    result.columns = ["BUREAU_" + _f for _f in result.columns]
    feature = pd.merge(feature, result.reset_index(), on="SK_ID_CURR", how="left")
    return feature
