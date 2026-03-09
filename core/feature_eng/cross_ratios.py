from __future__ import annotations

import gc

import pandas as pd


def association_perc(feature):
    amt_annuity = [_f for _f in feature.columns if 'AMT_ANNUITY' in _f and "application_AMT_ANNUITY" not in _f]
    amt_goods_price = [_f for _f in feature.columns if
                       'AMT_GOODS_PRICE' in _f and "application_AMT_GOODS_PRICE" not in _f]
    amt_credit = [_f for _f in feature.columns if 'AMT_CREDIT' in _f and "application_AMT_CREDIT" not in _f]
    annuity_credit_perc = [_f for _f in feature.columns if
                           'ANNUITY_CREDIT_PERC' in _f and "application_ANNUITY_CREDIT_PERC" not in _f]
    appr_process_start = [_f for _f in feature.columns if
                          'HOUR_APPR_PROCESS_START' in _f and "application_HOUR_APPR_PROCESS_START" not in _f]

    for f_ in amt_annuity[1:]:
        feature[f_ + "_application_AMT_ANNUITY_PERC"] = (feature[f_] / feature["application_AMT_ANNUITY"]).astype(
            'float32')
    for f_ in amt_goods_price[1:]:
        feature[f_ + "_application_AMT_GOODS_PRICE_PERC"] = (
                    feature[f_] / feature["application_AMT_GOODS_PRICE"]).astype('float32')
    for f_ in amt_credit[1:]:
        feature[f_ + "_application_AMT_CREDIT_PERC"] = (feature[f_] / feature["application_AMT_CREDIT"]).astype(
            'float32')
    for f_ in annuity_credit_perc[1:]:
        feature[f_ + "_application_ANNUITY_CREDIT_PERC_PERC"] = (
                    feature[f_] / feature["application_ANNUITY_CREDIT_PERC"]).astype('float32')
    for f_ in appr_process_start[1:]:
        feature[f_ + "_application_HOUR_APPR_PROCESS_START_PERC"] = (
                    feature[f_] / feature["application_HOUR_APPR_PROCESS_START"]).astype('float32')

    feature["POS_CNT_INSTALMENT_FUTURE_1MON3MON_PERC"] = feature["POS_CNT_INSTALMENT_FUTURE_MEAN_2MON"] / feature[
        "POS_CNT_INSTALMENT_FUTURE_MEAN_4MON"]
    feature["POS_CNT_INSTALMENT_FUTURE_3MON12MON_PERC"] = feature["POS_CNT_INSTALMENT_FUTURE_MEAN_4MON"] / feature[
        "POS_CNT_INSTALMENT_FUTURE_MEAN_12MON"]
    feature["INSTAL_DPD_BOOL_120d365DAY_PERC"] = feature["INSTAL_120DAY_DPD_BOOL_MEAN"] / feature[
        "INSTAL_365DAY_DPD_BOOL_MEAN"]
    feature["INSTAL_DPD_120d365_PERC"] = feature["INSTAL_120DAY_DPD_MAX"] / feature["INSTAL_365DAY_DPD_MAX"]
    feature["INSTAL_DPD_120dALL_PERC"] = feature["INSTAL_120DAY_DPD_MAX"] / feature["INSTAL_DPD_MAX"]
    feature["INSTAL_DPD_30d365_PERC"] = feature["INSTAL_30DAY_DPD_MAX"] / feature["INSTAL_365DAY_DPD_MAX"]
    feature["INSTAL_DPD_30d120_PERC"] = feature["INSTAL_30DAY_DPD_MAX"] / feature["INSTAL_120DAY_DPD_MAX"]
    feature["INSTAL_DPD_30dALL_PERC"] = feature["INSTAL_30DAY_DPD_MAX"] / feature["INSTAL_DPD_MAX"]
    feature["INSTAL_ENTRY_PAYMENT_3T10_PERC"] = feature["INSTAL_3TIMES_DAYS_ENTRY_PAYMENT_STD"] / feature[
        "INSTAL_10TIMES_DAYS_ENTRY_PAYMENT_STD"]
    gc.collect()
    return feature
