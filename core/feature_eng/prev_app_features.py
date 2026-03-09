from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from core.feature_eng.app_features import one_hot_encoder

DATA_PATH = "dataset/"


def previous_applications(feature):
    prev = pd.read_csv(DATA_PATH + 'prior_apps.csv')

    ins_target_path = DATA_PATH + "INS_TARGET.csv"
    if os.path.exists(ins_target_path):
        ins_target = pd.read_csv(ins_target_path)
        ins_target = ins_target.groupby(["SK_ID_PREV"], as_index=False)["DPD_BOOL"].mean()
    else:
        ins_target = pd.DataFrame(columns=["SK_ID_PREV", "DPD_BOOL"])

    cc_target_path = DATA_PATH + "CC_TARGET.csv"
    if os.path.exists(cc_target_path):
        cc_target = pd.read_csv(cc_target_path)
        cc_target = cc_target.groupby(["SK_ID_PREV"], as_index=False)["SK_DPD_DIFF"].mean()
    else:
        cc_target = pd.DataFrame(columns=["SK_ID_PREV", "SK_DPD_DIFF"])

    pos_target_path = DATA_PATH + "P0S_TARGET.csv"
    if os.path.exists(pos_target_path):
        pos_target = pd.read_csv(pos_target_path)
        pos_target = pos_target.groupby(["SK_ID_PREV"], as_index=False)["POS_TARGET"].mean()
    else:
        pos_target = pd.DataFrame(columns=["SK_ID_PREV", "POS_TARGET"])

    prev = pd.merge(prev, ins_target, on="SK_ID_PREV", how='left')
    prev = pd.merge(prev, cc_target, on="SK_ID_PREV", how='left')
    prev = pd.merge(prev, pos_target, on="SK_ID_PREV", how='left')

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_CREDIT'] + 1)
    prev['APP_COODS_PERC'] = prev['AMT_APPLICATION'] / (prev['AMT_GOODS_PRICE'] + 1)
    prev['ANNUITY_CREDIT_PERC'] = prev['AMT_ANNUITY'] / (prev['AMT_CREDIT'] + 1)
    prev['GOODS_PRICE_CREDIT_PERC'] = prev['AMT_CREDIT'] / (prev['AMT_GOODS_PRICE'] + 1)

    num_aggregations = {
        'AMT_ANNUITY': ["first", "sum", 'max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean', 'sum', "std"],
        'AMT_CREDIT': ["first", "sum", 'max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean', 'sum', "std"],
        'ANNUITY_CREDIT_PERC': ['max', 'mean', 'sum', "std"],
        'GOODS_PRICE_CREDIT_PERC': ['max', 'mean', 'sum', "std"],
        'AMT_DOWN_PAYMENT': ['max', 'mean', 'sum', "std"],
        'AMT_GOODS_PRICE': ["first", "std", 'max', 'mean', "sum"],
        'HOUR_APPR_PROCESS_START': ['max', 'mean', "sum"],
        'RATE_DOWN_PAYMENT': ['mean', 'sum', "std"],
        'DAYS_DECISION': ['mean', 'sum', "std"],
        'CNT_PAYMENT': ['mean', 'sum', "std"],
        'APP_COODS_PERC': ['mean', 'sum', "std"],
    }

    for bin_feature in ["CODE_REJECT_REASON", "NAME_CASH_LOAN_PURPOSE", "NAME_GOODS_CATEGORY", "PRODUCT_COMBINATION"]:
        for item in ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "DPD_BOOL", "SK_DPD_DIFF",
                      "POS_TARGET"]:
            temp = prev.groupby(bin_feature)[item].mean()
            prev[bin_feature + "_" + item] = prev[bin_feature].map(temp).astype('float32')
            num_aggregations[bin_feature + "_" + item] = ['mean', 'max', "std"]

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=False)

    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg(dict(num_aggregations, **cat_aggregations)).astype('float32')
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, approved_agg, right_index=True, left_index=True, how="left")

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, refused_agg, right_index=True, left_index=True, how="left")

    lastdate = prev[prev['DAYS_DECISION'] >= -30]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['30DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = prev[prev['DAYS_DECISION'] >= -90]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['90DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = prev[prev['DAYS_DECISION'] >= -120]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['120DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = approved[approved['DAYS_DECISION'] >= -365]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(num_aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(['ONEYEAR_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    prev_agg = pd.merge(prev_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    del refused, refused_agg, approved, approved_agg, prev, lastdate_agg, lastdate

    feature = pd.merge(feature, prev_agg.reset_index(), on="SK_ID_CURR", how="left")
    return feature
