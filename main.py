import numpy as np
import pandas as pd
import gc
from Models import LightGBM, CatBoost, LinearRegression
from Dataset import KaggleDataset

# %% Load dataset and preprocess
DATA_PATH = './data/'
DEBUG = False

dataset = KaggleDataset(DATA_PATH, debug=DEBUG)
dataset.load_data()

#%%
X, Y = dataset.get_train_data()
X_test = dataset.get_test_data()

# %% Split to train and val data
gc.collect()
RANDOM_SEED = 143
NFOLD = 5
BAGGING = False
PREDICT_TEST = False
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost, CatBoost or LSTM


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(
        objective='binary',
        metric='auc',
        boosting='gbdt',
        num_leaves=32, lr=0.02., bagging_fraction=0.8715623,
        max_depth=8,
        max_bin=255,
        feature_fraction=0.9497036, bagging_freq=3,
        # min_data_in_leaf=12,
        use_missing=True, zero_as_missing=False,
        min_split_gain=0.0222415,
        min_child_weight=40,
        lambda_l1=0.04, lambda_l2=0.073,
        device='gpu', num_threads=11)

    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)

    model = LightGBM(**LightGBM_params)


if PREDICT_TEST:
    out = model.cv_predict(X, Y, X_test,
                                      logloss=True,
                                      **fit_params)

else:
    out = model.cv(X, Y, **fit_params)
