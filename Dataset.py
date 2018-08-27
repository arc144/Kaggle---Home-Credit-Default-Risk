import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import warnings
import feature_extraction as fe
warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def one_hot_encoder(df, nan_as_category=True):
    # One-hot encoding for categorical columns with get_dummies
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


class KaggleDataset():
    '''Dataset class to load and process competition data'''

    def __init__(self, data_path, debug=False, num_workers=4):
        self.data_path = data_path
        self.debug = debug
        self.num_workers = num_workers

    def load_data(self, use_application_agg=False,
                  use_diffs_only_aggregates=True,
                  remove_useless=True):
        num_rows = 10000 if self.debug else None
        df = self.application_train_test(num_rows)
        # Preprocess
        with timer("Process bureau and bureau_balance"):
            bureau = self.bureau_and_balance(num_rows)
            print("Bureau df shape:", bureau.shape)
            df = df.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with timer("Process previous_applications"):
            df = self.previous_applications(df, num_rows)
            gc.collect()
        with timer("Process POS-CASH balance"):
            df = self.pos_cash(df, num_rows)
            gc.collect()
        with timer("Process installments payments"):
            df = self.installments_payments(df, num_rows)
            gc.collect()
        with timer("Process credit card balance"):
            df = self.credit_card_balance(df, num_rows)
            gc.collect()
        # if use_application_agg:
        #     with timer("Process applications aggregations"):
        #         app_agg = self.application_aggregations(
        #             num_rows, use_diffs_only=use_diffs_only_aggregates)
        #         print("app_agg df shape:", app_agg.shape)
        #         df = df.join(app_agg, how='left', on='SK_ID_CURR')
        #     del app_agg
        #     gc.collect()

        self.train_df = df[df['TARGET'].notnull()]
        self.test_df = df[df['TARGET'].isnull()]
        del df
        gc.collect()
        # Define feature names
        if remove_useless:
            useless_feats = self.get_useless_features()
        else:
            useless_feats = []
        self.feats = [f for f in self.train_df.columns if f not in useless_feats + [
            'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        print('Using a total of {} train features'.format(len(self.feats)))

    def get_train_data(self):
        # Get train data as array
        x = self.train_df[self.feats].values
        y = self.train_df['TARGET'].values
        return x, y

    def get_test_data(self):
        # Get test data as array
        x = self.test_df[self.feats].values
        return x

    def application_train_test(self, num_rows=None, nan_as_category=False):
        # Preprocess application_train.csv and application_test.csv
        # Read data and merge
        df = pd.read_csv(
            '{}/application_train.csv'.format(self.data_path), nrows=num_rows)
        test_df = pd.read_csv(
            '{}/application_test.csv'.format(self.data_path), nrows=num_rows)
        print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
        df = df.append(test_df).reset_index()
        # DATA CLEANING
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) & (
            'FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

        inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
            'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df[
            'AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = df[
            'AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
        df['NEW_INC_PER_CHLD'] = df[
            'AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
        df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df[
            'DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df[
            'AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * \
            df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df[
            'AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        # ############################ New FE ############################
        df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['external_sources_weighted'] = df.EXT_SOURCE_1 * \
            2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
        df['cnt_non_child'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
        df['child_to_non_child_ratio'] = df[
            'CNT_CHILDREN'] / df['cnt_non_child']
        df['income_per_non_child'] = df[
            'AMT_INCOME_TOTAL'] / df['cnt_non_child']
        df['credit_per_person'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
        df['credit_per_child'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
        df['credit_per_non_child'] = df['AMT_CREDIT'] / df['cnt_non_child']
        for function_name in ['min', 'max', 'sum', 'mean', 'std', 'nanmedian']:
            df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        df['short_employment'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)
        df['young_age'] = (df['DAYS_BIRTH'] < -14000).astype(int)
        # ##############################################################
        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        # Categorical features with One-Hot encode
        df, cat_cols = one_hot_encoder(df, nan_as_category)
        dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                     'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                     'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                     'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                     'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                     'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                     'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        df = df.drop(dropcolum, axis=1)
        del test_df
        gc.collect()
        return df

    def bureau_and_balance(self, num_rows=None, nan_as_category=True,
                           fill_missing=True, fill_value=0):
        # Preprocess bureau.csv and bureau_balance.csv
        bureau = pd.read_csv(
            '{}/bureau.csv'.format(self.data_path), nrows=num_rows)
        bb = pd.read_csv(
            '{}/bureau_balance.csv'.format(self.data_path), nrows=num_rows)

        # Clean data
        bureau['DAYS_CREDIT_ENDDATE'][
            bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
        bureau['DAYS_CREDIT_UPDATE'][
            bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
        bureau['DAYS_ENDDATE_FACT'][
            bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

        if fill_missing:
            bureau['AMT_CREDIT_SUM'].fillna(fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_DEBT'].fillna(fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(fill_value, inplace=True)
            bureau['CNT_CREDIT_PROLONG'].fillna(fill_value, inplace=True)

        bb, bb_cat = one_hot_encoder(bb, nan_as_category)
        bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                                   for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg
        gc.collect()

        bureau['bureau_credit_enddate_binary'] = (
            bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var', 'count'],
            'bureau_credit_enddate_binary': ['mean'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }

        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat:
            cat_aggregations[cat] = ['mean']
        for cat in bb_cat:
            cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(
            ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(
            ['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(
            ['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()

        # New feats
        for status in ['ACTIVE_', 'CLOSED_']:
            bureau_agg['{}bureau_debt_credit_ratio'.format(status)] = \
                bureau_agg['{}AMT_CREDIT_SUM_DEBT_SUM'.format(status)] / \
                bureau_agg['{}AMT_CREDIT_SUM_SUM'.format(status)]

            bureau_agg['{}bureau_overdue_debt_ratio'.format(status)] = \
                bureau_agg['{}AMT_CREDIT_SUM_OVERDUE_SUM'.format(status)] / \
                bureau_agg['{}AMT_CREDIT_SUM_DEBT_SUM'.format(status)]
        return bureau_agg

    def previous_applications(self, df, num_rows=None, nan_as_category=True):
        # Preprocess previous_applications.csv
        prev = pd.read_csv(
            '{}/previous_application.csv'.format(self.data_path), nrows=num_rows)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        # Add Feature engineer
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        ####################################
        prev_fe = fe.PreviousApplicationFeatures(num_workers=self.num_workers)
        prev_fe = prev_fe.fit(prev)
        prev_fe.persist('prev_fe')
        gm = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = gm.transform(df, prev_fe.features)
        ########################################
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'APP_CREDIT_PERC': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(
            ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        # Previous Applications: Approved Applications - only numerical
        # features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(
            ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(
            ['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del approved, approved_agg, refused, refused_agg
        gc.collect()
        # join
        df = df.join(prev_agg, how='left', on='SK_ID_CURR')
        print("Previous applications df shape:",
              prev.shape[1] + prev_fe.features.shape[1])
        return df

    def pos_cash(self, df, num_rows=None, nan_as_category=True):
        # Preprocess POS_CASH_balance.csv
        pos = pd.read_csv(
            '{}/POS_CASH_balance.csv'.format(self.data_path), nrows=num_rows)
        #################
        pos_fe = fe.POSCASHBalanceFeatures(
            last_k_agg_periods=[6, 12],
            last_k_trend_periods=[6, 12, 30],
            num_workers=self.num_workers)
        pos_fe.load(
            '{}/pos_cash_balance_features.joblib'.format(self.data_path))
        # pos_fe = pos_fe.fit(pos)
        # pos_fe.persist('pos_fe')
        gm = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = gm.transform(df, pos_fe.features)
        ###################
        pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(
            ['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        print("Pos-cash balance df shape:",
              pos_fe.features.shape[1] + pos_agg.shape[1])
        # join with main df
        df = df.join(pos_agg, how='left', on='SK_ID_CURR')
        del pos, pos_agg, pos_fe
        gc.collect()
        return df

    def installments_payments(self, df, num_rows=None, nan_as_category=True):
        # Preprocess installments_payments.csv
        ins = pd.read_csv(
            '{}/installments_payments.csv'.format(self.data_path), nrows=num_rows)
        #################
        ins_fe = fe.InstallmentPaymentsFeatures(
            last_k_agg_periods=[1, 5, 10, 20, 50, 100],
            last_k_agg_period_fractions=[
                (5, 20), (5, 50), (10, 50), (10, 100), (20, 100)],
            last_k_trend_periods=[10, 50, 100, 500],
            num_workers=self.num_workers)
        # ins_fe = ins_fe.fit(ins)
        # ins_fe.persist('ins_fe')
        ins_fe.load(
            '{}/installments_features.joblib'.format(self.data_path))
        gm = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = gm.transform(df, ins_fe.features)
        ###################
        ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
        # Percentage and difference paid in each installment (amount paid and
        # installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['mean',  'var'],
            'PAYMENT_DIFF': ['mean', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(
            ['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        print("Installments payments df shape:",
              ins_fe.features.shape[1] + ins_agg.shape[1])
        # join with main df
        df = df.join(ins_agg, how='left', on='SK_ID_CURR')
        return df

    def credit_card_balance(self, df, num_rows=None, nan_as_category=True):
        # Preprocess credit_card_balance.csv
        cc = pd.read_csv(
            '{}/credit_card_balance.csv'.format(self.data_path), nrows=num_rows)
        cc['AMT_DRAWINGS_ATM_CURRENT'][
            cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
        cc['AMT_DRAWINGS_CURRENT'][cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
        ####################################
        cc_fe = fe.CreditCardBalanceFeatures(num_workers=self.num_workers)
        cc_fe = cc_fe.fit(cc)
        cc_fe.persist('cc_fe')
        gm = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = gm.transform(df, cc_fe.features)
        ########################################
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(
            ['max', 'min', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                                   for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        print("Credit card balance df shape:",
              cc_agg.shape[1] + cc_fe.features.shape[1])
        df = df.join(cc_agg, how='left', on='SK_ID_CURR')
        del cc, cc_agg, cc_fe
        gc.collect()
        return df

    def application_aggregations(self,  num_rows=None, use_diffs_only=True):
        df = pd.read_csv(
            '{}/application_train.csv'.format(self.data_path), nrows=num_rows)
        test_df = pd.read_csv(
            '{}/application_test.csv'.format(self.data_path), nrows=num_rows)
        df = df.append(test_df).reset_index()
        # DATA CLEANING
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

        groupby_aggregations = self.get_application_aggregation_recipes()

        # Groupby
        features = []
        groupby_feature_names = []
        for groupby_cols, specs in groupby_aggregations:
            group_object = df.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(
                    groupby_cols, select, agg)
                group_features = group_object[select].agg(agg).reset_index() \
                    .rename(index=str,
                            columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]]

                features.append((groupby_cols, group_features))
                groupby_feature_names.append(groupby_aggregate_name)

        # Merge
        for groupby_cols, groupby_features in features:
            df = df.merge(groupby_features,
                          on=groupby_cols,
                          how='left')
        # Diff
        diff_feature_names = []
        for groupby_cols, specs in groupby_aggregations:
            for select, agg in specs:
                if agg in ['mean', 'median', 'max', 'min']:
                    groupby_aggregate_name = self._create_colname_from_specs(
                        groupby_cols, select, agg)
                    diff_feature_name = '{}_diff'.format(
                        groupby_aggregate_name)
                    abs_diff_feature_name = '{}_abs_diff'.format(
                        groupby_aggregate_name)

                    df[diff_feature_name] = df[
                        select] - df[groupby_aggregate_name]
                    df[abs_diff_feature_name] = np.abs(
                        df[select] - df[groupby_aggregate_name])

                    diff_feature_names.append(diff_feature_name)
                    diff_feature_names.append(abs_diff_feature_name)

        if use_diffs_only:
            feature_names = diff_feature_names
        else:
            feature_names = groupby_feature_names + diff_feature_names

        return df[feature_names].astype(np.float32)

    def get_application_aggregation_recipes(self):
        cols_to_agg = ['AMT_CREDIT',
                       'AMT_ANNUITY',
                       'AMT_INCOME_TOTAL',
                       'AMT_GOODS_PRICE',
                       'EXT_SOURCE_1',
                       'EXT_SOURCE_2',
                       'EXT_SOURCE_3',
                       'OWN_CAR_AGE',
                       'REGION_POPULATION_RELATIVE',
                       'DAYS_REGISTRATION',
                       'CNT_CHILDREN',
                       'CNT_FAM_MEMBERS',
                       'DAYS_ID_PUBLISH',
                       'DAYS_BIRTH',
                       'DAYS_EMPLOYED'
                       ]

        aggs = ['min', 'mean', 'max', 'sum', 'var']
        aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

        APPLICATION_AGGREGATION_RECIPIES = [
            (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
            (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                    ('AMT_INCOME_TOTAL', 'mean'),
                                                    ('DAYS_REGISTRATION', 'mean'),
                                                    ('EXT_SOURCE_1', 'mean')]),
            (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                         ('CNT_CHILDREN', 'mean'),
                                                         ('DAYS_ID_PUBLISH', 'mean')]),
            (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                                   ('EXT_SOURCE_2', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                          ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                          ('APARTMENTS_AVG', 'mean'),
                                                          ('BASEMENTAREA_AVG',
                                                           'mean'),
                                                          ('EXT_SOURCE_1', 'mean'),
                                                          ('EXT_SOURCE_2', 'mean'),
                                                          ('EXT_SOURCE_3', 'mean'),
                                                          ('NONLIVINGAREA_AVG',
                                                           'mean'),
                                                          ('OWN_CAR_AGE', 'mean'),
                                                          ('YEARS_BUILD_AVG', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                    ('EXT_SOURCE_1', 'mean')]),
            (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                   ('CNT_CHILDREN', 'mean'),
                                   ('CNT_FAM_MEMBERS', 'mean'),
                                   ('DAYS_BIRTH', 'mean'),
                                   ('DAYS_EMPLOYED', 'mean'),
                                   ('DAYS_ID_PUBLISH', 'mean'),
                                   ('DAYS_REGISTRATION', 'mean'),
                                   ('EXT_SOURCE_1', 'mean'),
                                   ('EXT_SOURCE_2', 'mean'),
                                   ('EXT_SOURCE_3', 'mean')])]
        return APPLICATION_AGGREGATION_RECIPIES

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)

    def get_useless_features(self):
        LOW_IMPORTANCE = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                          'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'ELEVATORS_AVG',
                          'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'FLAG_CONT_MOBILE',
                          'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                          'LANDAREA_MEDI', 'LANDAREA_MODE', 'REG_CITY_NOT_WORK_CITY',
                          'NEW_LIVE_IND_SUM', 'children_ratio', 'cnt_non_child',
                          'child_to_non_child_ratio', 'short_employment', 'young_age',
                          'EMERGENCYSTATE_MODE_No', 'EMERGENCYSTATE_MODE_Yes',
                          'FONDKAPREMONT_MODE_not specified',
                          'FONDKAPREMONT_MODE_reg oper account',
                          'HOUSETYPE_MODE_block of flats', 'NAME_CONTRACT_TYPE_Cash loans',
                          'NAME_CONTRACT_TYPE_Revolving loans',
                          'NAME_EDUCATION_TYPE_Academic degree',
                          'NAME_FAMILY_STATUS_Single / not married',
                          'NAME_HOUSING_TYPE_Co-op apartment',
                          'NAME_HOUSING_TYPE_Rented apartment',
                          'NAME_HOUSING_TYPE_With parents', 'NAME_INCOME_TYPE_Businessman',
                          'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner',
                          'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Unemployed',
                          'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
                          'NAME_TYPE_SUITE_Unaccompanied', 'OCCUPATION_TYPE_Cleaning staff',
                          'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_HR staff',
                          'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Managers',
                          'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
                          'OCCUPATION_TYPE_Security staff', 'ORGANIZATION_TYPE_Advertising',
                          'ORGANIZATION_TYPE_Agriculture',
                          'ORGANIZATION_TYPE_Business Entity Type 2',
                          'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Culture',
                          'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government',
                          'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 1',
                          'ORGANIZATION_TYPE_Industry: type 10',
                          'ORGANIZATION_TYPE_Industry: type 11',
                          'ORGANIZATION_TYPE_Industry: type 12',
                          'ORGANIZATION_TYPE_Industry: type 13',
                          'ORGANIZATION_TYPE_Industry: type 2',
                          'ORGANIZATION_TYPE_Industry: type 4',
                          'ORGANIZATION_TYPE_Industry: type 5',
                          'ORGANIZATION_TYPE_Industry: type 6',
                          'ORGANIZATION_TYPE_Industry: type 8',
                          'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Legal Services',
                          'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Postal',
                          'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion',
                          'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Telecom',
                          'ORGANIZATION_TYPE_Trade: type 1',
                          'ORGANIZATION_TYPE_Trade: type 2',
                          'ORGANIZATION_TYPE_Trade: type 4',
                          'ORGANIZATION_TYPE_Trade: type 5',
                          'ORGANIZATION_TYPE_Trade: type 6',
                          'ORGANIZATION_TYPE_Trade: type 7',
                          'ORGANIZATION_TYPE_Transport: type 1',
                          'ORGANIZATION_TYPE_Transport: type 2',
                          'ORGANIZATION_TYPE_Transport: type 4',
                          'ORGANIZATION_TYPE_University', 'WALLSMATERIAL_MODE_Block',
                          'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Wooden',
                          'WEEKDAY_APPR_PROCESS_START_FRIDAY',
                          'WEEKDAY_APPR_PROCESS_START_THURSDAY',
                          'BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
                          'BURO_CREDIT_ACTIVE_nan_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 1_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 2_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 3_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 4_MEAN',
                          'BURO_CREDIT_CURRENCY_nan_MEAN',
                          'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
                          'BURO_CREDIT_TYPE_Interbank credit_MEAN',
                          'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
                          'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
                          'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
                          'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
                          'BURO_CREDIT_TYPE_Real estate loan_MEAN',
                          'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
                          'BURO_CREDIT_TYPE_nan_MEAN', 'BURO_STATUS_nan_MEAN_MEAN',
                          'CLOSED_CREDIT_DAY_OVERDUE_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_OVERDUE_SUM',
                          'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM', 'CLOSED_MONTHS_BALANCE_MAX_MAX',
                          'CLOSED_bureau_overdue_debt_ratio',
                          'previous_application_prev_was_approved',
                          'previous_application_prev_was_refused',
                          'prev_applications_prev_was_revolving_loan',
                          'PREV_AMT_GOODS_PRICE_MAX', 'PREV_RATE_DOWN_PAYMENT_MAX',
                          'PREV_DAYS_DECISION_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_nan_MEAN',
                          'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN',
                          'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
                          'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN',
                          'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
                          'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
                          'PREV_CODE_REJECT_REASON_CLIENT_MEAN',
                          'PREV_CODE_REJECT_REASON_SCOFR_MEAN',
                          'PREV_CODE_REJECT_REASON_SYSTEM_MEAN',
                          'PREV_CODE_REJECT_REASON_XAP_MEAN',
                          'PREV_CODE_REJECT_REASON_nan_MEAN',
                          'PREV_NAME_TYPE_SUITE_Family_MEAN',
                          'PREV_NAME_TYPE_SUITE_Group of people_MEAN',
                          'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
                          'PREV_NAME_CLIENT_TYPE_nan_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Education_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
                          'PREV_NAME_PORTFOLIO_Cars_MEAN', 'PREV_NAME_PORTFOLIO_POS_MEAN',
                          'PREV_NAME_PORTFOLIO_nan_MEAN',
                          'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
                          'PREV_NAME_PRODUCT_TYPE_nan_MEAN',
                          'PREV_CHANNEL_TYPE_Car dealer_MEAN',
                          'PREV_CHANNEL_TYPE_Country-wide_MEAN',
                          'PREV_CHANNEL_TYPE_nan_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
                          'PREV_NAME_YIELD_GROUP_middle_MEAN',
                          'PREV_NAME_YIELD_GROUP_nan_MEAN',
                          'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_nan_MEAN',
                          'APPROVED_AMT_DOWN_PAYMENT_MEAN', 'APPROVED_RATE_DOWN_PAYMENT_MAX',
                          'REFUSED_AMT_APPLICATION_MEAN', 'REFUSED_AMT_CREDIT_MAX',
                          'REFUSED_AMT_CREDIT_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Demand_MEAN',
                          'POS_NAME_CONTRACT_STATUS_XNA_MEAN',
                          'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'credit_card_number_of_loans',
                          'credit_card_average_of_days_past_due',
                          'credit_card_total_installments', 'CC_MONTHS_BALANCE_MAX',
                          'CC_MONTHS_BALANCE_MEAN', 'CC_AMT_BALANCE_MIN',
                          'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX',
                          'CC_AMT_DRAWINGS_ATM_CURRENT_MIN',
                          'CC_AMT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_AMT_DRAWINGS_CURRENT_MEAN',
                          'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
                          'CC_AMT_INST_MIN_REGULARITY_MIN',
                          'CC_AMT_INST_MIN_REGULARITY_MEAN',
                          'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX',
                          'CC_AMT_RECEIVABLE_PRINCIPAL_MAX', 'CC_AMT_RECIVABLE_MAX',
                          'CC_AMT_RECIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_MAX',
                          'CC_AMT_TOTAL_RECEIVABLE_SUM', 'CC_CNT_DRAWINGS_ATM_CURRENT_MIN',
                          'CC_CNT_DRAWINGS_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX',
                          'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
                          'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
                          'CC_CNT_INSTALMENT_MATURE_CUM_VAR', 'CC_SK_DPD_MAX',
                          'CC_SK_DPD_MIN', 'CC_SK_DPD_MEAN', 'CC_SK_DPD_VAR',
                          'CC_SK_DPD_DEF_MIN', 'CC_SK_DPD_DEF_VAR',
                          'CC_number_of_installments_SUM',
                          'CC_credit_card_max_loading_of_credit_limit_MAX',
                          'CC_credit_card_max_loading_of_credit_limit_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Active_MAX',
                          'CC_NAME_CONTRACT_STATUS_Active_MIN',
                          'CC_NAME_CONTRACT_STATUS_Active_SUM',
                          'CC_NAME_CONTRACT_STATUS_Approved_MAX',
                          'CC_NAME_CONTRACT_STATUS_Approved_MIN',
                          'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Approved_SUM',
                          'CC_NAME_CONTRACT_STATUS_Approved_VAR',
                          'CC_NAME_CONTRACT_STATUS_Completed_MAX',
                          'CC_NAME_CONTRACT_STATUS_Completed_MIN',
                          'CC_NAME_CONTRACT_STATUS_Completed_SUM',
                          'CC_NAME_CONTRACT_STATUS_Completed_VAR',
                          'CC_NAME_CONTRACT_STATUS_Demand_MAX',
                          'CC_NAME_CONTRACT_STATUS_Demand_MIN',
                          'CC_NAME_CONTRACT_STATUS_Demand_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Demand_SUM',
                          'CC_NAME_CONTRACT_STATUS_Demand_VAR',
                          'CC_NAME_CONTRACT_STATUS_Refused_MAX',
                          'CC_NAME_CONTRACT_STATUS_Refused_MIN',
                          'CC_NAME_CONTRACT_STATUS_Refused_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Refused_SUM',
                          'CC_NAME_CONTRACT_STATUS_Refused_VAR',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
                          'CC_NAME_CONTRACT_STATUS_Signed_MAX',
                          'CC_NAME_CONTRACT_STATUS_Signed_MIN',
                          'CC_NAME_CONTRACT_STATUS_Signed_SUM',
                          'CC_NAME_CONTRACT_STATUS_nan_MAX',
                          'CC_NAME_CONTRACT_STATUS_nan_MIN',
                          'CC_NAME_CONTRACT_STATUS_nan_MEAN',
                          'CC_NAME_CONTRACT_STATUS_nan_SUM',
                          'CC_NAME_CONTRACT_STATUS_nan_VAR', 'CC_COUNT']

        USELESS_COLUMNS = ['FLAG_DOCUMENT_10',
                           'FLAG_DOCUMENT_12',
                           'FLAG_DOCUMENT_13',
                           'FLAG_DOCUMENT_14',
                           'FLAG_DOCUMENT_15',
                           'FLAG_DOCUMENT_16',
                           'FLAG_DOCUMENT_17',
                           'FLAG_DOCUMENT_19',
                           'FLAG_DOCUMENT_2',
                           'FLAG_DOCUMENT_20',
                           'FLAG_DOCUMENT_21']

        HIGHLY_CORRELATED_NUMERICAL_COLUMNS = ['AMT_GOODS_PRICE',
                                               'APARTMENTS_MEDI',
                                               'APARTMENTS_MODE',
                                               'BASEMENTAREA_MEDI',
                                               'BASEMENTAREA_MODE',
                                               'COMMONAREA_MEDI',
                                               'COMMONAREA_MODE',
                                               'ELEVATORS_MEDI',
                                               'ELEVATORS_MODE',
                                               'ENTRANCES_MEDI',
                                               'ENTRANCES_MODE',
                                               'FLAG_EMP_PHONE',
                                               'FLOORSMAX_MEDI',
                                               'FLOORSMAX_MODE',
                                               'FLOORSMIN_MEDI',
                                               'FLOORSMIN_MODE',
                                               'LANDAREA_MEDI',
                                               'LANDAREA_MODE',
                                               'LIVINGAPARTMENTS_MEDI',
                                               'LIVINGAPARTMENTS_MODE',
                                               'LIVINGAREA_MEDI',
                                               'LIVINGAREA_MODE',
                                               'NONLIVINGAPARTMENTS_MEDI',
                                               'NONLIVINGAPARTMENTS_MODE',
                                               'NONLIVINGAREA_MEDI',
                                               'NONLIVINGAREA_MODE',
                                               'OBS_60_CNT_SOCIAL_CIRCLE',
                                               'REGION_RATING_CLIENT_W_CITY',
                                               'YEARS_BEGINEXPLUATATION_MEDI',
                                               'YEARS_BEGINEXPLUATATION_MODE',
                                               'YEARS_BUILD_MEDI',
                                               'YEARS_BUILD_MODE']
        return LOW_IMPORTANCE + USELESS_COLUMNS + HIGHLY_CORRELATED_NUMERICAL_COLUMNS


class KaggleDataset_Sol5():
    '''Dataset class to load and process competition data'''
    ID_COLUMNS = ['SK_ID_CURR']
    TARGET_COLUMNS = ['TARGET']
    CATEGORICAL_COLUMNS = ['CODE_GENDER', 'EMERGENCYSTATE_MODE', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_18', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FONDKAPREMONT_MODE', 'HOUR_APPR_PROCESS_START', 'HOUSETYPE_MODE', 'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION', 'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START']  # noqa 501
    NUMERICAL_COLUMNS = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'COMMONAREA_AVG', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION',
                         'DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OWN_CAR_AGE', 'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'TOTALAREA_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG']  # noqa501

    def __init__(self, data_path, debug=False, num_workers=4, fill_missing=True, fill_value=0, use_diffs_only=True):
        self.data_path = data_path
        self.debug = debug
        self.num_workers = num_workers
        self.fill_missing = fill_missing
        self.fill_value = fill_value
        self.use_diffs_only = use_diffs_only
        self.get_application_aggregation_recipes()

    def load_data(self, remove_useless=False):
        num_rows = 10000 if self.debug else None
        df = self.application_train_test(num_rows)
        # Preprocess
        with timer("Process bureau and bureau_balance"):
            df = self.bureau_and_balance(df, num_rows)
            gc.collect()
        with timer("Process previous_applications"):
            df = self.previous_applications(df, num_rows)
            gc.collect()
        with timer("Process POS-CASH balance"):
            df = self.pos_cash(df, num_rows)
            gc.collect()
        with timer("Process installments payments"):
            df = self.installments_payments(df, num_rows)
            gc.collect()
        with timer("Process credit card balance"):
            df = self.credit_card_balance(df, num_rows)
            gc.collect()
        with timer("Process app agg"):
            df = self.application_agg(df, num_rows)
            gc.collect()
        with timer("Process bureau agg"):
            df = self.bureau_agg(df, num_rows)
            gc.collect()
        with timer("Process previous agg"):
            df = self.previous_agg(df, num_rows)
            gc.collect()
        with timer("Process pos_cash agg"):
            df = self.pos_cash_agg(df, num_rows)
            gc.collect()
        with timer("Process installments_payments agg"):
            df = self.installments_payments_agg(df, num_rows)
            gc.collect()
        with timer("Process credit_card agg"):
            df = self.credit_card_agg(df, num_rows)
            gc.collect()
        with timer("Process categorical features"):
            df = self.categorical(df, num_rows)
            gc.collect()

        # Divide df into train and test
        self.train_df = df[df['TARGET'].notnull()]
        self.test_df = df[df['TARGET'].isnull()]
        print("Train shape: {}, test shape: {}".format(
            self.train_df.shape, self.test_df.shape))
        del df
        gc.collect()
        # Define feature names
        if remove_useless:
            useless_feats = self.get_useless_features()
        else:
            useless_feats = []
        self.feats = [f for f in self.train_df.columns if f not in useless_feats + [
            'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
        print('Using a total of {} train features'.format(len(self.feats)))

    def get_train_data(self):
        # Get train data as array
        x = self.train_df[self.feats].values
        y = self.train_df['TARGET'].values
        return x, y

    def get_test_data(self):
        # Get test data as array
        x = self.test_df[self.feats].values
        return x

    def application_train_test(self, num_rows=None, nan_as_category=False):
        # Preprocess application_train.csv and application_test.csv
        # Read data and merge
        df = pd.read_csv(
            '{}/application_train.csv'.format(self.data_path), nrows=num_rows)
        test_df = pd.read_csv(
            '{}/application_test.csv'.format(self.data_path), nrows=num_rows)
        print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
        df = df.append(test_df).reset_index()
        cleaner = fe.ApplicationCleaning()
        df = cleaner.transform(df)

        extractor = fe.ApplicationFeatures(
            self.CATEGORICAL_COLUMNS, self.NUMERICAL_COLUMNS)
        df = extractor.transform(df)
        del test_df
        gc.collect()
        return df

    def application_agg(self, curr_df, num_rows=None, nan_as_category=False):
        # Preprocess application_train.csv and application_test.csv
        # Read data and merge
        df = pd.read_csv(
            '{}/application_train.csv'.format(self.data_path), nrows=num_rows)
        test_df = pd.read_csv(
            '{}/application_test.csv'.format(self.data_path), nrows=num_rows)
        df = df.append(test_df).reset_index()
        cleaner = fe.ApplicationCleaning()
        df = cleaner.transform(df)

        extractor = fe.GroupbyAggregateDiffs(
            self.APPLICATION_AGGREGATION_RECIPIES, use_diffs_only=self.use_diffs_only)
        df = extractor.fit(df).transform(df)
        df = curr_df.merge(df, on='SK_ID_CURR', how='left')

        return df

    def bureau_and_balance(self, df, num_rows=None, nan_as_category=True,
                           fill_missing=True, fill_value=0):
        # Preprocess bureau.csv and bureau_balance.csv
        bureau = pd.read_csv(
            '{}/bureau.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.BureauCleaning(self.fill_missing, self.fill_value)
        bureau = cleaner.transform(bureau)
        extractor = fe.BureauFeatures()
        bureau_df = extractor.fit(bureau).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, bureau_df)
        return df

    def bureau_agg(self, df, num_rows=None, nan_as_category=True,
                   fill_missing=True, fill_value=0):
        # Preprocess bureau.csv and bureau_balance.csv
        bureau = pd.read_csv(
            '{}/bureau.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.BureauCleaning(self.fill_missing, self.fill_value)
        bureau = cleaner.transform(bureau)
        extractor = fe.GroupbyAggregate(
            'bureau', ('SK_ID_CURR', 'SK_ID_CURR'), self.BUREAU_AGGREGATION_RECIPIES)
        bureau_df = extractor.fit(bureau).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, bureau_df)
        return df

    def previous_applications(self, df, num_rows=None, nan_as_category=True):
        # Preprocess previous_applications.csv
        prev = pd.read_csv(
            '{}/previous_application.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.PreviousApplicationCleaning()
        prev = cleaner.transform(prev)

        extractor = fe.PreviousApplicationFeatures(
            numbers_of_applications=[1, 2, 3, 4, 5])
        prev_df = extractor.fit(prev).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, prev_df)
        return df

    def previous_agg(self, df, num_rows=None, nan_as_category=True):
        # Preprocess previous_applications.csv
        prev = pd.read_csv(
            '{}/previous_application.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.PreviousApplicationCleaning()
        prev = cleaner.transform(prev)

        extractor = fe.GroupbyAggregate(
            'prev', ('SK_ID_CURR', 'SK_ID_CURR'), self.PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)
        prev_df = extractor.fit(prev).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, prev_df)
        return df

    def pos_cash(self, df, num_rows=None, nan_as_category=True):
        # Preprocess POS_CASH_balance.csv
        pos = pd.read_csv(
            '{}/POS_CASH_balance.csv'.format(self.data_path), nrows=num_rows)
        #################
        pos_fe = fe.POSCASHBalanceFeatures(
            last_k_agg_periods=[6, 12],
            last_k_trend_periods=[6, 12, 30],
            num_workers=self.num_workers)
        pos_fe.load(
            '{}/pos_cash_balance_features.joblib'.format(self.data_path))
        # pos_fe = pos_fe.fit(pos)
        # pos_fe.persist('pos_fe')
        merger = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, pos_fe.features)
        return df

    def pos_cash_agg(self, df, num_rows=None, nan_as_category=True):
        # Preprocess POS_CASH_balance.csv
        pos = pd.read_csv(
            '{}/POS_CASH_balance.csv'.format(self.data_path), nrows=num_rows)
        #################
        extractor = fe.GroupbyAggregate(
            'pos_cash', ('SK_ID_CURR', 'SK_ID_CURR'), self.POS_CASH_BALANCE_AGGREGATION_RECIPIES)
        prev_df = extractor.fit(pos).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, prev_df)
        return df

    def installments_payments(self, df, num_rows=None, nan_as_category=True):
        # Preprocess installments_payments.csv
        ins = pd.read_csv(
            '{}/installments_payments.csv'.format(self.data_path), nrows=num_rows)
        #################
        ins_fe = fe.InstallmentPaymentsFeatures(
            last_k_agg_periods=[1, 5, 10, 20, 50, 100],
            last_k_agg_period_fractions=[
                (5, 20), (5, 50), (10, 50), (10, 100), (20, 100)],
            last_k_trend_periods=[10, 50, 100, 500],
            num_workers=self.num_workers)
        # ins_fe = ins_fe.fit(ins)
        # ins_fe.persist('ins_fe')
        ins_fe.load(
            '{}/installments_features.joblib'.format(self.data_path))
        merger = fe.GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, ins_fe.features)
        return df

    def installments_payments_agg(self, df, num_rows=None, nan_as_category=True):
        # Preprocess installments_payments.csv
        ins = pd.read_csv(
            '{}/installments_payments.csv'.format(self.data_path), nrows=num_rows)
        #################
        extractor = fe.GroupbyAggregate(
            'installments', ('SK_ID_CURR', 'SK_ID_CURR'), self.INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)
        prev_df = extractor.fit(ins).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, prev_df)
        return df

    def credit_card_balance(self, df, num_rows=None, nan_as_category=True):
        # Preprocess credit_card_balance.csv
        cc = pd.read_csv(
            '{}/credit_card_balance.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.CreditCardCleaning()
        cc = cleaner.transform(cc)

        extractor = fe.CreditCardBalanceFeatures()
        cc = extractor.fit(cc)

        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, cc.features)
        return df

    def credit_card_agg(self, df, num_rows=None, nan_as_category=True):
        # Preprocess credit_card_balance.csv
        cc = pd.read_csv(
            '{}/credit_card_balance.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.CreditCardCleaning()
        cc = cleaner.transform(cc)

        extractor = fe.GroupbyAggregate(
            'credit_card', ('SK_ID_CURR', 'SK_ID_CURR'), self.CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)
        prev_df = extractor.fit(cc).features
        merger = fe.GroupbyMerge(('SK_ID_CURR', 'SK_ID_CURR'))
        df = merger.transform(df, prev_df)
        return df

    def categorical(self, curr_df, num_rows):
        df = pd.read_csv(
            '{}/application_train.csv'.format(self.data_path), nrows=num_rows)
        test_df = pd.read_csv(
            '{}/application_test.csv'.format(self.data_path), nrows=num_rows)

        cleaner = fe.ApplicationCleaning()
        df = cleaner.transform(df)
        test_df = cleaner.transform(test_df)

        encoder = fe.CategoricalEncoder(
            categorical_columns=self.CATEGORICAL_COLUMNS)
        encoder.fit(df.drop('TARGET', axis=1), df['TARGET'])
        df = encoder.transform(df)
        test_df = encoder.transform(test_df)
        df = df.append(test_df).reset_index()

        df = curr_df.merge(df, on='SK_ID_CURR', how='left')
        return df

    def get_application_aggregation_recipes(self):
        cols_to_agg = ['AMT_CREDIT',
                       'AMT_ANNUITY',
                       'AMT_INCOME_TOTAL',
                       'AMT_GOODS_PRICE',
                       'EXT_SOURCE_1',
                       'EXT_SOURCE_2',
                       'EXT_SOURCE_3',
                       'OWN_CAR_AGE',
                       'REGION_POPULATION_RELATIVE',
                       'DAYS_REGISTRATION',
                       'CNT_CHILDREN',
                       'CNT_FAM_MEMBERS',
                       'DAYS_ID_PUBLISH',
                       'DAYS_BIRTH',
                       'DAYS_EMPLOYED'
                       ]
        aggs = ['min', 'mean', 'max', 'sum', 'var']
        aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

        self.APPLICATION_AGGREGATION_RECIPIES = [
            (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
            (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
            (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                    ('AMT_INCOME_TOTAL', 'mean'),
                                                    ('DAYS_REGISTRATION', 'mean'),
                                                    ('EXT_SOURCE_1', 'mean')]),
            (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                         ('CNT_CHILDREN', 'mean'),
                                                         ('DAYS_ID_PUBLISH', 'mean')]),
            (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                                   ('EXT_SOURCE_2', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                          ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                          ('APARTMENTS_AVG', 'mean'),
                                                          ('BASEMENTAREA_AVG',
                                                           'mean'),
                                                          ('EXT_SOURCE_1', 'mean'),
                                                          ('EXT_SOURCE_2', 'mean'),
                                                          ('EXT_SOURCE_3', 'mean'),
                                                          ('NONLIVINGAREA_AVG',
                                                           'mean'),
                                                          ('OWN_CAR_AGE', 'mean'),
                                                          ('YEARS_BUILD_AVG', 'mean')]),
            (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                    ('EXT_SOURCE_1', 'mean')]),
            (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                   ('CNT_CHILDREN', 'mean'),
                                   ('CNT_FAM_MEMBERS', 'mean'),
                                   ('DAYS_BIRTH', 'mean'),
                                   ('DAYS_EMPLOYED', 'mean'),
                                   ('DAYS_ID_PUBLISH', 'mean'),
                                   ('DAYS_REGISTRATION', 'mean'),
                                   ('EXT_SOURCE_1', 'mean'),
                                   ('EXT_SOURCE_2', 'mean'),
                                   ('EXT_SOURCE_3', 'mean')]),
        ]

        self.BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'),
                                            ('CREDIT_ACTIVE', 'size')
                                            ]
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_ANNUITY',
                           'AMT_CREDIT_SUM',
                           'AMT_CREDIT_SUM_DEBT',
                           'AMT_CREDIT_SUM_LIMIT',
                           'AMT_CREDIT_SUM_OVERDUE',
                           'AMT_CREDIT_MAX_OVERDUE',
                           'CNT_CREDIT_PROLONG',
                           'CREDIT_DAY_OVERDUE',
                           'DAYS_CREDIT',
                           'DAYS_CREDIT_ENDDATE',
                           'DAYS_CREDIT_UPDATE'
                           ]:
                self.BUREAU_AGGREGATION_RECIPIES.append((select, agg))
        self.BUREAU_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], self.BUREAU_AGGREGATION_RECIPIES)]

        self.CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_BALANCE',
                           'AMT_CREDIT_LIMIT_ACTUAL',
                           'AMT_DRAWINGS_ATM_CURRENT',
                           'AMT_DRAWINGS_CURRENT',
                           'AMT_DRAWINGS_OTHER_CURRENT',
                           'AMT_DRAWINGS_POS_CURRENT',
                           'AMT_PAYMENT_CURRENT',
                           'CNT_DRAWINGS_ATM_CURRENT',
                           'CNT_DRAWINGS_CURRENT',
                           'CNT_DRAWINGS_OTHER_CURRENT',
                           'CNT_INSTALMENT_MATURE_CUM',
                           'MONTHS_BALANCE',
                           'SK_DPD',
                           'SK_DPD_DEF'
                           ]:
                self.CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append(
                    (select, agg))
        self.CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], self.CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]

        self.INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_INSTALMENT',
                           'AMT_PAYMENT',
                           'DAYS_ENTRY_PAYMENT',
                           'DAYS_INSTALMENT',
                           'NUM_INSTALMENT_NUMBER',
                           'NUM_INSTALMENT_VERSION'
                           ]:
                self.INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append(
                    (select, agg))
        self.INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], self.INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

        self.POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['MONTHS_BALANCE',
                           'SK_DPD',
                           'SK_DPD_DEF'
                           ]:
                self.POS_CASH_BALANCE_AGGREGATION_RECIPIES.append(
                    (select, agg))
        self.POS_CASH_BALANCE_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], self.POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

        self.PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
        for agg in ['mean', 'min', 'max', 'sum', 'var']:
            for select in ['AMT_ANNUITY',
                           'AMT_APPLICATION',
                           'AMT_CREDIT',
                           'AMT_DOWN_PAYMENT',
                           'AMT_GOODS_PRICE',
                           'CNT_PAYMENT',
                           'DAYS_DECISION',
                           'HOUR_APPR_PROCESS_START',
                           'RATE_DOWN_PAYMENT'
                           ]:
                self.PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append(
                    (select, agg))
        self.PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [
            (['SK_ID_CURR'], self.PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)

    def get_useless_features(self):
        LOW_IMPORTANCE = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                          'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'ELEVATORS_AVG',
                          'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'FLAG_CONT_MOBILE',
                          'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                          'LANDAREA_MEDI', 'LANDAREA_MODE', 'REG_CITY_NOT_WORK_CITY',
                          'NEW_LIVE_IND_SUM', 'children_ratio', 'cnt_non_child',
                          'child_to_non_child_ratio', 'short_employment', 'young_age',
                          'EMERGENCYSTATE_MODE_No', 'EMERGENCYSTATE_MODE_Yes',
                          'FONDKAPREMONT_MODE_not specified',
                          'FONDKAPREMONT_MODE_reg oper account',
                          'HOUSETYPE_MODE_block of flats', 'NAME_CONTRACT_TYPE_Cash loans',
                          'NAME_CONTRACT_TYPE_Revolving loans',
                          'NAME_EDUCATION_TYPE_Academic degree',
                          'NAME_FAMILY_STATUS_Single / not married',
                          'NAME_HOUSING_TYPE_Co-op apartment',
                          'NAME_HOUSING_TYPE_Rented apartment',
                          'NAME_HOUSING_TYPE_With parents', 'NAME_INCOME_TYPE_Businessman',
                          'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner',
                          'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Unemployed',
                          'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people',
                          'NAME_TYPE_SUITE_Unaccompanied', 'OCCUPATION_TYPE_Cleaning staff',
                          'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_HR staff',
                          'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Managers',
                          'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Secretaries',
                          'OCCUPATION_TYPE_Security staff', 'ORGANIZATION_TYPE_Advertising',
                          'ORGANIZATION_TYPE_Agriculture',
                          'ORGANIZATION_TYPE_Business Entity Type 2',
                          'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Culture',
                          'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government',
                          'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 1',
                          'ORGANIZATION_TYPE_Industry: type 10',
                          'ORGANIZATION_TYPE_Industry: type 11',
                          'ORGANIZATION_TYPE_Industry: type 12',
                          'ORGANIZATION_TYPE_Industry: type 13',
                          'ORGANIZATION_TYPE_Industry: type 2',
                          'ORGANIZATION_TYPE_Industry: type 4',
                          'ORGANIZATION_TYPE_Industry: type 5',
                          'ORGANIZATION_TYPE_Industry: type 6',
                          'ORGANIZATION_TYPE_Industry: type 8',
                          'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Legal Services',
                          'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Postal',
                          'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion',
                          'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Telecom',
                          'ORGANIZATION_TYPE_Trade: type 1',
                          'ORGANIZATION_TYPE_Trade: type 2',
                          'ORGANIZATION_TYPE_Trade: type 4',
                          'ORGANIZATION_TYPE_Trade: type 5',
                          'ORGANIZATION_TYPE_Trade: type 6',
                          'ORGANIZATION_TYPE_Trade: type 7',
                          'ORGANIZATION_TYPE_Transport: type 1',
                          'ORGANIZATION_TYPE_Transport: type 2',
                          'ORGANIZATION_TYPE_Transport: type 4',
                          'ORGANIZATION_TYPE_University', 'WALLSMATERIAL_MODE_Block',
                          'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Wooden',
                          'WEEKDAY_APPR_PROCESS_START_FRIDAY',
                          'WEEKDAY_APPR_PROCESS_START_THURSDAY',
                          'BURO_CREDIT_DAY_OVERDUE_MEAN', 'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
                          'BURO_CREDIT_ACTIVE_nan_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 1_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 2_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 3_MEAN',
                          'BURO_CREDIT_CURRENCY_currency 4_MEAN',
                          'BURO_CREDIT_CURRENCY_nan_MEAN',
                          'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
                          'BURO_CREDIT_TYPE_Interbank credit_MEAN',
                          'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
                          'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
                          'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
                          'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
                          'BURO_CREDIT_TYPE_Real estate loan_MEAN',
                          'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
                          'BURO_CREDIT_TYPE_nan_MEAN', 'BURO_STATUS_nan_MEAN_MEAN',
                          'CLOSED_CREDIT_DAY_OVERDUE_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_OVERDUE_SUM',
                          'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN',
                          'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM', 'CLOSED_MONTHS_BALANCE_MAX_MAX',
                          'CLOSED_bureau_overdue_debt_ratio',
                          'previous_application_prev_was_approved',
                          'previous_application_prev_was_refused',
                          'prev_applications_prev_was_revolving_loan',
                          'PREV_AMT_GOODS_PRICE_MAX', 'PREV_RATE_DOWN_PAYMENT_MAX',
                          'PREV_DAYS_DECISION_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
                          'PREV_NAME_CONTRACT_TYPE_nan_MEAN',
                          'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN',
                          'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
                          'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN',
                          'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN',
                          'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN',
                          'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
                          'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
                          'PREV_CODE_REJECT_REASON_CLIENT_MEAN',
                          'PREV_CODE_REJECT_REASON_SCOFR_MEAN',
                          'PREV_CODE_REJECT_REASON_SYSTEM_MEAN',
                          'PREV_CODE_REJECT_REASON_XAP_MEAN',
                          'PREV_CODE_REJECT_REASON_nan_MEAN',
                          'PREV_NAME_TYPE_SUITE_Family_MEAN',
                          'PREV_NAME_TYPE_SUITE_Group of people_MEAN',
                          'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
                          'PREV_NAME_CLIENT_TYPE_nan_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Education_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN',
                          'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
                          'PREV_NAME_PORTFOLIO_Cars_MEAN', 'PREV_NAME_PORTFOLIO_POS_MEAN',
                          'PREV_NAME_PORTFOLIO_nan_MEAN',
                          'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
                          'PREV_NAME_PRODUCT_TYPE_nan_MEAN',
                          'PREV_CHANNEL_TYPE_Car dealer_MEAN',
                          'PREV_CHANNEL_TYPE_Country-wide_MEAN',
                          'PREV_CHANNEL_TYPE_nan_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN',
                          'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
                          'PREV_NAME_YIELD_GROUP_middle_MEAN',
                          'PREV_NAME_YIELD_GROUP_nan_MEAN',
                          'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
                          'PREV_PRODUCT_COMBINATION_nan_MEAN',
                          'APPROVED_AMT_DOWN_PAYMENT_MEAN', 'APPROVED_RATE_DOWN_PAYMENT_MAX',
                          'REFUSED_AMT_APPLICATION_MEAN', 'REFUSED_AMT_CREDIT_MAX',
                          'REFUSED_AMT_CREDIT_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
                          'POS_NAME_CONTRACT_STATUS_Demand_MEAN',
                          'POS_NAME_CONTRACT_STATUS_XNA_MEAN',
                          'POS_NAME_CONTRACT_STATUS_nan_MEAN', 'credit_card_number_of_loans',
                          'credit_card_average_of_days_past_due',
                          'credit_card_total_installments', 'CC_MONTHS_BALANCE_MAX',
                          'CC_MONTHS_BALANCE_MEAN', 'CC_AMT_BALANCE_MIN',
                          'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX',
                          'CC_AMT_DRAWINGS_ATM_CURRENT_MIN',
                          'CC_AMT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_AMT_DRAWINGS_CURRENT_MEAN',
                          'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN',
                          'CC_AMT_INST_MIN_REGULARITY_MIN',
                          'CC_AMT_INST_MIN_REGULARITY_MEAN',
                          'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX',
                          'CC_AMT_RECEIVABLE_PRINCIPAL_MAX', 'CC_AMT_RECIVABLE_MAX',
                          'CC_AMT_RECIVABLE_SUM', 'CC_AMT_TOTAL_RECEIVABLE_MAX',
                          'CC_AMT_TOTAL_RECEIVABLE_SUM', 'CC_CNT_DRAWINGS_ATM_CURRENT_MIN',
                          'CC_CNT_DRAWINGS_CURRENT_MIN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX',
                          'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
                          'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
                          'CC_CNT_INSTALMENT_MATURE_CUM_VAR', 'CC_SK_DPD_MAX',
                          'CC_SK_DPD_MIN', 'CC_SK_DPD_MEAN', 'CC_SK_DPD_VAR',
                          'CC_SK_DPD_DEF_MIN', 'CC_SK_DPD_DEF_VAR',
                          'CC_number_of_installments_SUM',
                          'CC_credit_card_max_loading_of_credit_limit_MAX',
                          'CC_credit_card_max_loading_of_credit_limit_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Active_MAX',
                          'CC_NAME_CONTRACT_STATUS_Active_MIN',
                          'CC_NAME_CONTRACT_STATUS_Active_SUM',
                          'CC_NAME_CONTRACT_STATUS_Approved_MAX',
                          'CC_NAME_CONTRACT_STATUS_Approved_MIN',
                          'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Approved_SUM',
                          'CC_NAME_CONTRACT_STATUS_Approved_VAR',
                          'CC_NAME_CONTRACT_STATUS_Completed_MAX',
                          'CC_NAME_CONTRACT_STATUS_Completed_MIN',
                          'CC_NAME_CONTRACT_STATUS_Completed_SUM',
                          'CC_NAME_CONTRACT_STATUS_Completed_VAR',
                          'CC_NAME_CONTRACT_STATUS_Demand_MAX',
                          'CC_NAME_CONTRACT_STATUS_Demand_MIN',
                          'CC_NAME_CONTRACT_STATUS_Demand_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Demand_SUM',
                          'CC_NAME_CONTRACT_STATUS_Demand_VAR',
                          'CC_NAME_CONTRACT_STATUS_Refused_MAX',
                          'CC_NAME_CONTRACT_STATUS_Refused_MIN',
                          'CC_NAME_CONTRACT_STATUS_Refused_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Refused_SUM',
                          'CC_NAME_CONTRACT_STATUS_Refused_VAR',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
                          'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
                          'CC_NAME_CONTRACT_STATUS_Signed_MAX',
                          'CC_NAME_CONTRACT_STATUS_Signed_MIN',
                          'CC_NAME_CONTRACT_STATUS_Signed_SUM',
                          'CC_NAME_CONTRACT_STATUS_nan_MAX',
                          'CC_NAME_CONTRACT_STATUS_nan_MIN',
                          'CC_NAME_CONTRACT_STATUS_nan_MEAN',
                          'CC_NAME_CONTRACT_STATUS_nan_SUM',
                          'CC_NAME_CONTRACT_STATUS_nan_VAR', 'CC_COUNT']

        USELESS_COLUMNS = ['FLAG_DOCUMENT_10',
                           'FLAG_DOCUMENT_12',
                           'FLAG_DOCUMENT_13',
                           'FLAG_DOCUMENT_14',
                           'FLAG_DOCUMENT_15',
                           'FLAG_DOCUMENT_16',
                           'FLAG_DOCUMENT_17',
                           'FLAG_DOCUMENT_19',
                           'FLAG_DOCUMENT_2',
                           'FLAG_DOCUMENT_20',
                           'FLAG_DOCUMENT_21']

        HIGHLY_CORRELATED_NUMERICAL_COLUMNS = ['AMT_GOODS_PRICE',
                                               'APARTMENTS_MEDI',
                                               'APARTMENTS_MODE',
                                               'BASEMENTAREA_MEDI',
                                               'BASEMENTAREA_MODE',
                                               'COMMONAREA_MEDI',
                                               'COMMONAREA_MODE',
                                               'ELEVATORS_MEDI',
                                               'ELEVATORS_MODE',
                                               'ENTRANCES_MEDI',
                                               'ENTRANCES_MODE',
                                               'FLAG_EMP_PHONE',
                                               'FLOORSMAX_MEDI',
                                               'FLOORSMAX_MODE',
                                               'FLOORSMIN_MEDI',
                                               'FLOORSMIN_MODE',
                                               'LANDAREA_MEDI',
                                               'LANDAREA_MODE',
                                               'LIVINGAPARTMENTS_MEDI',
                                               'LIVINGAPARTMENTS_MODE',
                                               'LIVINGAREA_MEDI',
                                               'LIVINGAREA_MODE',
                                               'NONLIVINGAPARTMENTS_MEDI',
                                               'NONLIVINGAPARTMENTS_MODE',
                                               'NONLIVINGAREA_MEDI',
                                               'NONLIVINGAREA_MODE',
                                               'OBS_60_CNT_SOCIAL_CIRCLE',
                                               'REGION_RATING_CLIENT_W_CITY',
                                               'YEARS_BEGINEXPLUATATION_MEDI',
                                               'YEARS_BEGINEXPLUATATION_MODE',
                                               'YEARS_BUILD_MEDI',
                                               'YEARS_BUILD_MODE']
        return USELESS_COLUMNS + HIGHLY_CORRELATED_NUMERICAL_COLUMNS
