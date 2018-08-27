import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def generate_bagging_splits(n_size, nfold, bagging_size_ratio=1, random_seed=143):
    '''Generate random bagging splits'''
    np.random.seed(random_seed)
    ref = range(n_size)
    out_size = int(bagging_size_ratio * n_size)

    splits = []
    for _ in range(nfold):
        t_index = np.random.randint(0, n_size, size=out_size)
        v_index = [j for j in ref if j not in t_index]
        splits.append((t_index, v_index))

    return splits


def normalized_accuracy(y_true, y_pred, threshold=0.5, verbose=True):
    y_pred_copy = y_pred.copy()
    y_pred_copy[y_pred_copy >= threshold] = 1
    y_pred_copy[y_pred_copy < threshold] = 0
    acc1 = np.mean(y_true[y_true == 1] == y_pred_copy[y_true == 1])
    acc0 = np.mean(y_true[y_true == 0] == y_pred_copy[y_true == 0])
    norm_acc = np.mean([acc0, acc1])
    if verbose:
        print('Accuracy class 0: ', acc0)
        print('Accuracy class 1: ', acc1)
        print('Normalized accuracy: ', norm_acc)
    return {'norm_acc': norm_acc, 'acc0': acc0, 'acc1': acc1}


class NonFittedError(Exception):

    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class LightGBM():
    '''Microsoft LightGBM class wrapper'''

    def __init__(self, objective='regression', metric='rmse',
                 num_leaves=40, lr=0.005, bagging_fraction=0.7,
                 feature_fraction=0.6, bagging_frequency=6, device='gpu',
                 **kwargs):
        self.params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "bagging_fraction": bagging_fraction,
            "feature_fraction": feature_fraction,
            "bagging_frequency": bagging_frequency,
            "bagging_seed": 42,
            "verbosity": -1,
            "seed": 42,
            "device": device,
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

        if self.params['metric'] in ['auc', 'binary_logloss', 'multi_logloss']:
            self.get_best_metric = max
            self.get_best_iter = np.argmax
        else:
            self.get_best_metric = min
            self.get_best_iter = np.argmin

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150, return_oof_pred=True, **kwargs):
        # Train LGB model
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        self.model = lgb.train(self.params, lgtrain,
                               num_boost_round=steps,
                               valid_sets=[lgtrain, lgval],
                               early_stopping_rounds=ES_rounds,
                               verbose_eval=verbose,
                               evals_result=evals_result)
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
        else:
            pred = None
        return evals_result, pred

    def cv(self, X, Y, test_X=None, nfold=5, ES_rounds=100, steps=5000,
           random_seed=143, logloss=False,
           bootstrap=False, bagging_size_ratio=1,
           splits=None, importance_type='split', use_SMOTE=False):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = StratifiedKFold(n_splits=nfold, shuffle=True,
                                     random_state=random_seed)
                splits = kf.split(X, y=Y)

        oof_results = []
        y_true = []
        feature_imp = []
        results = []
        pred_y = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            if use_SMOTE:
                # print('Using SMOTE. Original Train size {}'.format(
                #     x_train.shape[0]))
                # sm = SMOTE(random_state=12, ratio='auto')
                # x_train, y_train = sm.fit_sample(x_train, y_train)
                # print('Resampled Train size {}'.format(
                #     x_train.shape[0]))
                idx = np.squeeze(np.where(y_train == 0))
                samp = np.random.choice(idx, size=idx.shape[0] // 3)
                samp = np.concatenate(
                    [samp, np.squeeze(np.where(y_train == 1))])
                np.random.shuffle(samp)
                x_train = x_train[samp]
                y_train = y_train[samp]

            with timer('Fold {} training'.format(i)):
                evals_result, oof_prediction = self.fit(train_X=x_train,
                                                        train_y=y_train,
                                                        val_X=x_val,
                                                        val_y=y_val,
                                                        ES_rounds=100,
                                                        steps=10000,
                                                        return_oof_pred=True)
            oof_results.extend(oof_prediction)
            y_true.extend(y_val)

            if evals_result:
                iter_nb = self.get_best_iter(evals_result['valid_1']['auc'])
                results.append(self.get_best_metric(
                    evals_result['valid_1']['auc']))
                feature_imp.append(self.model.feature_importance(
                    importance_type='split', iteration=iter_nb))
            normalized_accuracy(np.array(y_true), np.array(oof_results))

            # Get predictions if test)X
            if test_X is not None:
                if not i:
                    pred_y = self.predict(test_X, logloss=logloss)
                else:
                    pred_y += self.predict(test_X, logloss=logloss)

        print('Mean val score: {}, std {}'.format(
            np.mean(results), np.std(results)))
        pred_y = pred_y / nfold
        out = dict(oof_pred=np.array(oof_results),
                   fi=feature_imp,
                   test_pred=pred_y,
                   y_true=y_true,
                   results=results)
        return out

    def bagging_over_seed(self, X, Y,
                          split_seeds=[142, 143, 144],
                          model_seeds=[41, 42, 43],
                          **kwargs):
        oof_preds = []
        y_true = []
        feature_imp = []
        pred_y = []
        results = []
        for split_seed, model_seed in zip(split_seeds, model_seeds):
            self.params['seed'] = model_seed
            kwargs['random_seed'] = split_seed
            out = self.cv(X, Y, **kwargs)
            oof_preds.append(out['oof_pred'])
            y_true.append(out['y_true'])
            feature_imp.append(out['fi'])
            pred_y.append(out['test_pred'])
            results.extend(out['results'])
        out = dict(oof_pred=np.array(oof_preds),
                   oof_pred_mean=np.mean(oof_preds, axis=0),
                   fi=np.array(feature_imp),
                   fi_mean=np.mean(feature_imp, axis=(0, 1)),
                   test_pred=np.array(pred_y),
                   test_pred_mean=np.mean(pred_y, axis=0),
                   y_true=np.array(y_true),
                   results=np.array(results))
        print('Final bagging mean val score: {}, std {}'.format(
            np.mean(results), np.std(results)))
        return out

    def multi_seed_cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100,
                              steps=5000,
                              random_seed=[143, 135, 138], logloss=True,
                              bootstrap=False, bagging_size_ratio=1):
        '''Perform cv_predict for multiple seeds and avg them'''
        for i, seed in enumerate(random_seed):
            if not i:
                pred = self.cv_predict(X, Y, test_X, nfold=nfold,
                                       ES_rounds=ES_rounds, steps=steps,
                                       random_seed=seed, logloss=logloss,
                                       bootstrap=bootstrap,
                                       bagging_size_ratio=bagging_size_ratio)
            else:
                pred += self.cv_predict(X, Y, test_X, nfold=nfold,
                                        ES_rounds=ES_rounds, steps=steps,
                                        random_seed=seed, logloss=logloss,
                                        bootstrap=bootstrap,
                                        bagging_size_ratio=bagging_size_ratio)

        return pred / len(random_seed)

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        pred_y = self.model.predict(
            test_X, num_iteration=self.model.best_iteration)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    logloss=True, return_oof_pred=False, **kwargs):
        evals_result, oof_pred = self.fit(
            train_X, train_y, val_X, val_y, return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y

    def optmize_hyperparams(self, param_grid, X, Y,
                            cv=4, scoring='neg_mean_squared_error',
                            verbose=1):
        '''Use GridSearchCV to optimize models params'''
        params = self.params
        params['learning_rate'] = 0.05
        params['n_estimators'] = 1000
        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**params),
                                param_grid=param_grid,
                                scoring=scoring,
                                n_jobs=1,
                                iid=False,
                                cv=4)
        gsearch1.fit(X, Y)
        scores = gsearch1.grid_scores_
        best_params = gsearch1.best_params_
        best_score = np.sqrt(-gsearch1.best_score_)
        if verbose > 0:
            if verbose > 1:
                print('Scores are: ', scores)
            print('Best params: ', best_params)
            print('Best score: ', best_score)


class LRegression():
    '''Linear regression class wrapper'''

    def __init__(self, normalize):
        self.normalize = normalize
        self.is_fitted = False
        self.model = LogisticRegression(normalize)

    def fit(self, train_X, train_y, val_X=None, val_y=None,
            verbose=150, return_oof_pred=True):
        self.model.fit(X=train_X, y=train_y)
        self.is_fitted = True
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
            oof_result = roc_auc_score(val_y, pred)
        else:
            pred = None
            oof_result = None
        return oof_result, pred

    def cv(self, X, Y, nfold=5, random_seed=143,
           bootstrap=False, bagging_size_ratio=1,
           shuffle=True, oof_pred=False, verbose=100):
        # Train LGB model using CV
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=shuffle,
                       random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                return_oof_pred=True)

            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
        if oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5,
                   random_seed=143, shuffle=True, oof_pred=False,
                   bootstrap=False, bagging_size_ratio=1,
                   logloss=False, verbose=100):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=True, random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                return_oof_pred=True)

            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        if oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        if not self.is_fitted:
            raise NonFittedError(('Model has not been fitted.',
                                  ' First fit the model before predicting.'))
        pred_y = self.model.predict(test_X)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    verbose=150, return_oof_pred=True, logloss=False):
        evals_result, oof_pred = self.fit(train_X, train_y, val_X, val_y,
                                          return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss=logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y
