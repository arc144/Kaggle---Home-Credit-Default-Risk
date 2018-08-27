import numpy as np
import pandas as pd
import os
from Dataset import KaggleDataset
from Models import LightGBM, LRegression
#
TRAIN_NAMES = [
        'mixed_model_7pt933.npy',
        'vanilla_open_sol5_7pt919.npy']
         

TEST_NAMES = [
              ]
     
TRAIN_PATH = './oof_pred/'
TEST_PATH = './Stacking/Test/'
TRAIN_LIST = [os.path.join(TRAIN_PATH, name) for name in TRAIN_NAMES]
#TEST_LIST = [os.path.join(TEST_PATH, name) for name in TEST_NAMES]

Y_TRAIN = './oof_pred/oof_y.npy'

LOAD_TEST = False

X = []
for i, p in enumerate(TRAIN_LIST):
    X.append(np.load(p))
X = np.rollaxis(np.array(X), axis=1)
y = np.load(Y_TRAIN)

if LOAD_TEST:
    X_test = load_submissions_as_data_for_ensembling(TEST_LIST)
    X_test = np.log1p(X_test)

# %% Split to train and val data
RANDOM_SEED = 143
NFOLD = 3
BAGGING = True
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost, CatBoost, LiRegression


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(boosting='gbdt',
                           objective='binary',
                           metric='auc',
                           num_leaves=5, lr=0.0039, bagging_fraction=0.6,
                           max_depth=1,
                           max_bin=201,
                           feature_fraction=0.6, bagging_freq=3,
                           min_data_in_leaf=50,
                           min_sum_hessian_in_leaf=10,
                           use_missing=True, zero_as_missing=False,
                           lambda_l1=10, lambda_l2=10,
                           device='gpu', num_threads=11)

    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)

    model = LightGBM(**LightGBM_params)


elif MODEL_TYPE == 'LRegression':
    LRegression_params = dict(normalize=True)
    
    fit_params = dict(nfold=NFOLD,
                      random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      verbose=100)
    
    model = LRegression(**LRegression_params)
    
if LOAD_TEST:
    pred = model.cv_predict(X, y, X_test, logloss=False, **fit_params)

else:
    pred = model.cv(X, y, **fit_params)


# %%Create submission file
if LOAD_TEST:
    SUB_NAME = 'sub_stacking'
    test_index = pd.read_csv(TEST_LIST[0]).ID
    create_submission_file(test_index, pred, '{}.csv'.format(SUB_NAME))
    if USE_LEAK:
        merge_leaky_and_ML_sub(LEAKY_SUB_NAME,
                               '{}.csv'.format(SUB_NAME),
                               '{}_with_leak.csv'.format(SUB_NAME))
