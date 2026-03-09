"""Global project configuration: LightGBM hyperparameters, random seed."""

import warnings

warnings.filterwarnings("ignore")

seed = 1024

params = [
    {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.01,
        'max_depth': 8,
        'num_leaves': 35,
        'feature_fraction': 0.125681745820782,
        'bagging_freq': 5,
        'min_split_gain': 0.0970905919552776,
        'min_child_weight': 9.42012323936088,
        'reg_alpha': 4.82988348810309,
        'reg_lambda': 4.23709841316042,
        'verbose': 1,
        'device_type': 'gpu',
        'tree_learner': 'data',
        'max_bin': 255,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    },
    {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.015,
        'max_depth': 8,
        'num_leaves': 35,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'bagging_freq': 5,
        'min_split_gain': 0.0222415,
        'min_child_weight': 39.3259775,
        'reg_alpha': 0.041545473,
        'reg_lambda': 0.0735294,
        'verbose': 1,
    },
    {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.02,
        'max_depth': 8,
        'num_leaves': 34,
        'feature_fraction': 0.125681745820782,
        'bagging_fraction': 0.967396467593587,
        'bagging_freq': 5,
        'min_split_gain': 0.0970905919552776,
        'min_child_weight': 9.42012323936088,
        'reg_alpha': 4.82988348810309,
        'reg_lambda': 4.23709841316042,
        'verbose': 1,
    },
]
