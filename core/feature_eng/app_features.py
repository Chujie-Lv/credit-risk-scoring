from __future__ import annotations

from typing import Tuple

import gc
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("dataset")


def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, list[str]]:
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def safe_div(a, b) -> float:
    try:
        return float(a) / float(b)
    except Exception:
        return 0.0


def application_train_test(num_rows: int | None = None, nan_as_category: bool = True) -> pd.DataFrame:
    """Build base application features from train/test tables."""
    train_path = DATA_PATH / "train_main.csv"
    test_path = DATA_PATH / "test_main.csv"

    df = pd.read_csv(train_path, nrows=num_rows)
    test_df = pd.read_csv(test_path, nrows=num_rows)

    df = pd.concat([df, test_df], axis=0, ignore_index=True)

    # 预处理
    df = df[df["CODE_GENDER"] != "XNA"]
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)
    df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan, inplace=True)
    df["ORGANIZATION_TYPE"].replace("XNA", np.nan, inplace=True)
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # 原始特征
    df["DAYS_EMPLOYED_AGE"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"])
    df["PHONE_TO_BIRTH_PERC"] = df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_BIRTH"])
    df["PHONE_TO_BIRTH_PERC"] = df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_EMPLOYED"])
    df["CAR_TO_BIRTH_PERC"] = df["OWN_CAR_AGE"] / (df["DAYS_BIRTH"] * 365)
    df["CAR_TO_EMPLOY_PERC"] = df["OWN_CAR_AGE"] / (df["DAYS_EMPLOYED"])
    df["DAYS_ID_PUBLISH_BIRTH_PERC"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"])
    df["DAYS_REGISTRATION_EMPLOYED_PERC"] = df["DAYS_REGISTRATION"] / (df["DAYS_EMPLOYED"])

    df["DAYS_REG_TO_BIRTH_PERC"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"])
    df["DAYS_ID_PUBLISH_TO_EMPLOYED_PERC"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_EMPLOYED"])

    df["OBS_30_SC_TO_BIRTH_PERC"] = df["OBS_30_CNT_SOCIAL_CIRCLE"] / (df["DAYS_BIRTH"])
    df["DEF_30_SC_TO_EMPLOY_PERC"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["DAYS_EMPLOYED"])

    df["AMT_REQ_CREDIT_YEAR_TO_BIRTH_PERC"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"] / (df["DAYS_BIRTH"])
    df["AMT_REQ_CREDIT_YEAR_TO_EMPLOY_PERC"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"] / (df["DAYS_EMPLOYED"])

    df["DEF_30B60_PERC"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["DEF_60_CNT_SOCIAL_CIRCLE"])
    df["OBS_30B60_PERC"] = df["OBS_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_60_CNT_SOCIAL_CIRCLE"])

    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"])
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"])
    df["ANNUITY_CREDIT_PERC"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"])

    df["BASEMENTAREA_FAM_PERC"] = df["BASEMENTAREA_AVG"] / (df["CNT_FAM_MEMBERS"])
    df["CHILDREN_FAM_PERC"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"])
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"])

    df["EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCES_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_SOURCES_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_SOURCES_PERC"] = df["EXT_SOURCES_MIN"] / df["EXT_SOURCES_MAX"]
    df["EXT_SOURCES_NULL_NUM"] = np.sum(
        df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull(), axis=1
    )

    df["SCORES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["SCORES_STD"] = df["SCORES_STD"].fillna(df["SCORES_STD"].mean())

    df["EXT_SOURCES_MEAN_BIRTH_PERC"] = df["EXT_SOURCES_MEAN"] / (df["DAYS_BIRTH"] + 1)
    df["EXT_SOURCES_MEAN_ANNUITY_PERC"] = df["EXT_SOURCES_MEAN"] / (df["AMT_ANNUITY"] + 1)
    df["EXT_SOURCES_MEAN_CREDIT_PERC"] = df["EXT_SOURCES_MEAN"] / (df["AMT_CREDIT"] + 1)

    df["EXT1_AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * df["EXT_SOURCES_MEAN"]
    df["EXT1_AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"] * df["EXT_SOURCES_MEAN"]

    df["CREDIT_GOODS_PERC"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"])
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"]
    df["CREDIT_PER_CHILD"] = df["AMT_CREDIT"] / (1 + df["CNT_CHILDREN"])

    df["LIVINGAREA_PERC"] = df["LIVINGAREA_AVG"] / (df["NONLIVINGAREA_AVG"])

    docs = [_f for _f in df.columns if "FLAG_DOC" in _f]
    live = [
        _f
        for _f in df.columns
        if ("FLAG_" in _f) and ("FLAG_DOC" not in _f) and ("_FLAG_" not in _f)
    ]
    df["NEW_DOC_IND_KURT"] = df[docs].kurt(axis=1, numeric_only=True)
    df["NEW_DOC_IND_SKEW"] = df[docs].skew(axis=1, numeric_only=True)
    df["NEW_DOC_IND_STD"] = df[docs].std(axis=1, numeric_only=True)
    df["NEW_DOC_IND_MEAN"] = df[docs].mean(axis=1, numeric_only=True)
    df["NEW_DOC_IND_NULL"] = np.sum(df[docs].isnull(), axis=1)

    df["NEW_LIVE_IND_KURT"] = df[live].kurt(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_SKEW"] = df[live].skew(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_STD"] = df[live].std(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_MEAN"] = df[live].mean(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_NULL"] = np.sum(df[live].isnull(), axis=1)

    df["DAYS_BIRTH"] = (df["DAYS_BIRTH"] / -365).astype(int)

    df["REGION_POPULATION_CNT_CHILDREN_MUL"] = (
        df["CNT_CHILDREN"] * df["REGION_POPULATION_RELATIVE"]
    )
    df["REGION_POPULATION_CNT_FAM_MEMBERS_MUL"] = (
        df["CNT_FAM_MEMBERS"] * df["REGION_POPULATION_RELATIVE"]
    )
    df["CHILDREN_REGION_RATING_MUL"] = df["CNT_CHILDREN"] * df["REGION_RATING_CLIENT"]
    df["CHILDREN_REGION_RATING_CLIENT_W_CITY_MUL"] = (
        df["CNT_CHILDREN"] * df["REGION_RATING_CLIENT_W_CITY"]
    )
    df["FAMILY_REGION_RATING_MUL"] = df["CNT_FAM_MEMBERS"] * df["REGION_RATING_CLIENT"]
    df["FAMILY_REGION_RATING_CLIENT_W_CITY_MUL"] = (
        df["CNT_FAM_MEMBERS"] * df["REGION_RATING_CLIENT_W_CITY"]
    )

    for bin_feature in [
        "NAME_EDUCATION_TYPE",
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_CONTRACT_TYPE",
    ]:
        for item in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
            temp = df.groupby(bin_feature)[item].median()
            df[bin_feature + "_" + item] = df[bin_feature].map(temp)

    dropcolum = [
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "FLAG_EMP_PHONE",
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
    ]
    df = df.drop([c for c in dropcolum if c in df.columns], axis=1)

    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    df, _ = one_hot_encoder(df, nan_as_category=False)
    df.columns = ["application_" + f_ for f_ in df.columns]
    df.rename(
        columns={"application_SK_ID_CURR": "SK_ID_CURR", "application_TARGET": "TARGET"},
        inplace=True,
    )

    del test_df
    gc.collect()
    return df


__all__ = ["one_hot_encoder", "safe_div", "application_train_test"]

