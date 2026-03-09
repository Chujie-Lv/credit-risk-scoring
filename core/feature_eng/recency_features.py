from __future__ import annotations

import pandas as pd

DATA_PATH = "dataset/"


def top_k_ins_by_prev(feature):
    ins = pd.read_csv(DATA_PATH + 'inst_records.csv', nrows=None)
    ins.sort_values(by=['SK_ID_CURR', 'DAYS_INSTALMENT'], ascending=False, inplace=True)
    df_temp = ins.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['DAYS_INSTALMENT'].max().reset_index().rename(
        columns={'DAYS_INSTALMENT': 'near_DAYS_INSTALMENT'}).astype('float32')
    ins = ins.merge(df_temp, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    ins['top_k_PREV'] = ins.groupby(['SK_ID_CURR'])['near_DAYS_INSTALMENT'].rank(method='dense',
                                                                                  ascending=False).astype('float32')
    del ins['near_DAYS_INSTALMENT']

    ins['instalment_paid_late_in_days'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['instalment_paid_late'] = (ins['instalment_paid_late_in_days'] > 0).astype(int)
    ins['instalment_paid_over_amount'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['instalment_paid_over'] = (ins['instalment_paid_over_amount'] > 0).astype(int)

    features = pd.DataFrame({'SK_ID_CURR': ins['SK_ID_CURR'].unique()})
    features = features.set_index('SK_ID_CURR')

    aggregations = {
        'SK_ID_PREV': ['count'],
        'NUM_INSTALMENT_VERSION': ['nunique', 'mean'],
        'NUM_INSTALMENT_NUMBER': ['mean', 'var', 'max'],
        'DAYS_INSTALMENT': ['min', 'mean', 'var', 'max'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'var'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'var'],
        'DAYS_ENTRY_PAYMENT': ['min', 'mean', 'var', 'max'],
        'instalment_paid_late_in_days': ['max', 'mean', 'sum', 'var'],
        'instalment_paid_over_amount': ['max', 'mean', 'sum', 'var'],
        'instalment_paid_late': ['mean'],
        'instalment_paid_over': ['mean'],
    }

    for top_k in [1, 2, 3]:
        ins_top_k = ins[ins.top_k_PREV == top_k]
        ins_agg = ins_top_k.groupby('SK_ID_CURR').agg(aggregations).astype('float32')
        ins_agg.columns = pd.Index(
            ['INS_top_' + str(top_k) + '_prev_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        features = features.merge(ins_agg, on='SK_ID_CURR', how='left')

    del features['INS_top_1_prev_DAYS_INSTALMENT_MAX']
    del features['INS_top_1_prev_DAYS_ENTRY_PAYMENT_MAX']

    feature = pd.merge(feature, features, on='SK_ID_CURR', how='left')
    return feature
