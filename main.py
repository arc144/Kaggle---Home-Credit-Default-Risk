import numpy as np
import pandas as pd
import gc
from Models import LightGBM
from Dataset import KaggleDataset, KaggleDataset_Sol5
from analysis import display_importances

# %% Load dataset and preprocess
DATA_PATH = './data/' 
DEBUG = False


dataset = KaggleDataset(DATA_PATH, debug=DEBUG, num_workers=11)
dataset.load_data(remove_useless=True)
gc.collect()
#%%
X, Y = dataset.get_train_data()
#X = np.nan_to_num(X)
X_test = dataset.get_test_data()

# %% Split to train and val data
gc.collect()
BAGGING_OVER_SEED = False

RANDOM_SEED = 143
NFOLD = 5
BAGGING = False
USE_SMOTE = False
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost, CatBoost or LSTM


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(
        objective='binary',
        metric='auc',
        boosting='gbdt',
        num_leaves=30, lr=0.02, bagging_fraction=1,
        max_depth=-1,
        max_bin=300,
        feature_fraction=0.05, bagging_freq=1,
         min_data_in_leaf=70,
#        scale_pos_weight=15,
        is_unbalance=False,
        use_missing=True, zero_as_missing=False,
        min_split_gain=0.5,
#        min_child_weight=39.3259775,
        lambda_l1=0, lambda_l2=100,
        device='cpu', num_threads=11)
    

    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      use_SMOTE=USE_SMOTE)

    model = LightGBM(**LightGBM_params)

if BAGGING_OVER_SEED:
    out = model.bagging_over_seed(X, Y, test_X=X_test,
                      logloss=False,
                      **fit_params)
    fi = 'fi_mean'
    test_pred = 'test_pred_mean'
else:
    out = model.cv(X, Y, test_X=X_test,
                          logloss=False,
                          **fit_params)
    fi = 'fi'
    test_pred = 'test_pred'

fi_df = display_importances(out[fi], dataset.feats, 40)
# %%
gc.collect()
submission_file_name = 'submission1.csv'
test_df = dataset.test_df.copy()
test_df['TARGET'] = out[test_pred]
test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
test_df[['SK_ID_CURR', 'TARGET']].to_csv(
    submission_file_name, index=False)