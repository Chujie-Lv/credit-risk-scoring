from __future__ import annotations

import numpy as np
import pandas as pd

from core.feature_eng.app_features import one_hot_encoder

DATA_PATH = "dataset/"


def installments_payments(feature):
    ins = pd.read_csv(DATA_PATH + 'inst_records.csv')
    ins = ins.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending=False)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    ins['PAYMENT_PERC'] = ins['AMT_INSTALMENT'] / ins['AMT_PAYMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DPD_BOOL'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['AMT_PAYMENT_DPD_PERC'] = ins["AMT_PAYMENT"] / ins["DPD"]
    ins['PAYMENT_DIFF_DPD_PRODUCT'] = ins["PAYMENT_DIFF"] * ins["DPD"] / 30

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique', "size"],
        'DPD': ['max', 'mean', 'sum', 'min', 'std'],
        'DPD_BOOL': ['max', 'mean', 'sum', 'first', 'last', 'std'],
        'PAYMENT_PERC': ['max', 'mean', 'min', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'min', 'std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std'],
        'AMT_PAYMENT_DPD_PERC': ['max', 'mean', 'sum', 'std'],
        'PAYMENT_DIFF_DPD_PRODUCT': ['max', 'mean', 'sum', 'std'],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    lastdate = ins[ins['DAYS_INSTALMENT'] >= -30]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_30DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = ins[ins['DAYS_INSTALMENT'] >= -90]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_90DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = ins[ins['DAYS_INSTALMENT'] >= -120]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_120DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = ins[ins['DAYS_INSTALMENT'] >= -365]
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_365DAY_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = ins.groupby(["SK_ID_CURR"]).head(3)
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_3TIMES_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    lastdate = ins.groupby(["SK_ID_CURR"]).head(10)
    lastdate_agg = lastdate.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
    lastdate_agg.columns = pd.Index(
        ['INSTAL_10TIMES_' + e[0] + "_" + e[1].upper() for e in lastdate_agg.columns.tolist()])
    ins_agg = pd.merge(ins_agg, lastdate_agg, right_index=True, left_index=True, how="left")

    feature = pd.merge(feature, ins_agg.reset_index(), on="SK_ID_CURR", how="left")

    # Payment diff stats within same SK_ID_PREV
    df = ins[["SK_ID_CURR", "SK_ID_PREV", "AMT_PAYMENT", "DAYS_INSTALMENT"]]
    temp = df.copy()
    temp = temp.diff()
    temp = temp[temp.SK_ID_PREV == 0]
    df = pd.concat([df[["SK_ID_CURR"]], temp[["AMT_PAYMENT"]]], axis=1)
    diff_agg = df.groupby('SK_ID_CURR').agg({'AMT_PAYMENT': ['max', "min", "std"]})
    diff_agg.columns = pd.Index(
        ['INS_DIFF_PREV_' + e[0] + "_" + e[1].upper() for e in diff_agg.columns.tolist()])
    feature = pd.merge(feature, diff_agg.reset_index(), on="SK_ID_CURR", how="left")

    # Payment diff stats across SK_ID_PREV
    df = ins[["SK_ID_CURR", "SK_ID_PREV", "AMT_PAYMENT", "DAYS_INSTALMENT"]]
    temp = df.copy()
    temp = temp.diff()
    temp = temp[temp.SK_ID_CURR == 0]
    df = pd.concat([df[["SK_ID_CURR"]], temp[["AMT_PAYMENT"]]], axis=1)
    diff_agg = df.groupby('SK_ID_CURR').agg({'AMT_PAYMENT': ['max', "min", "std"]})
    diff_agg.columns = pd.Index(
        ['INS_DIFF_CURR_' + e[0] + "_" + e[1].upper() for e in diff_agg.columns.tolist()])
    feature = pd.merge(feature, diff_agg.reset_index(), on="SK_ID_CURR", how="left")
    return feature
