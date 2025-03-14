import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import warnings
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

    def __init__(self, data_path, debug=False):
        self.data_path = data_path
        self.debug = debug

    def load_data(self):
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
            prev = self.previous_applications(num_rows)
            print("Previous applications df shape:", prev.shape)
            df = df.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with timer("Process POS-CASH balance"):
            pos = self.pos_cash(num_rows)
            print("Pos-cash balance df shape:", pos.shape)
            df = df.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with timer("Process installments payments"):
            ins = self.installments_payments(num_rows)
            print("Installments payments df shape:", ins.shape)
            df = df.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with timer("Process credit card balance"):
            cc = self.credit_card_balance(num_rows)
            print("Credit card balance df shape:", cc.shape)
            df = df.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()
        # Divide df into train and test
        self.train_df = df[df['TARGET'].notnull()]
        self.test_df = df[df['TARGET'].isnull()]
        print("Train shape: {}, test shape: {}".format(
            self.train_df.shape, self.test_df.shape))
        del df
        gc.collect()
        # Define feature names
        self.feats = [f for f in self.train_df.columns if f not in [
            'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

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
        # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
        df = df[df['CODE_GENDER'] != 'XNA']

        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) & (
            'FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

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
        df['NEW_EXT_SOURCES_MEAN'] = df[
            ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_SCORES_STD'] = df[
            ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(
            df['NEW_SCORES_STD'].mean())
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df[
            'AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

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

    def bureau_and_balance(self, num_rows=None, nan_as_category=True):
        # Preprocess bureau.csv and bureau_balance.csv
        bureau = pd.read_csv(
            '{}/bureau.csv'.format(self.data_path), nrows=num_rows)
        bb = pd.read_csv(
            '{}/bureau_balance.csv'.format(self.data_path), nrows=num_rows)
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

        bureau['bureau_credit_active_binary'] = (
            bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
        bureau['bureau_credit_enddate_binary'] = (
            bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        bureau['bureau_average_of_past_loans_per_type'] = \
            bureau['bureau_number_of_past_loans'] / \
            bureau['bureau_number_of_loan_types']

        bureau['bureau_debt_credit_ratio'] = \
            bureau['bureau_total_customer_debt'] / \
            bureau['bureau_total_customer_credit']

        bureau['bureau_overdue_debt_ratio'] = \
            bureau['bureau_total_customer_overdue'] / \
            bureau['bureau_total_customer_debt']
        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var', 'count'],
            'CREDIT_TYPE': ['nunique'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'bureau_credit_active_binary': ['mean'],
            'bureau_credit_enddate_binary': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
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
        return bureau_agg

    def previous_applications(self, num_rows=None, nan_as_category=True):
        # Preprocess previous_applications.csv
        prev = pd.read_csv(
            '{}/previous_application.csv'.format(self.data_path), nrows=num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
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
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg

    def pos_cash(self, num_rows=None, nan_as_category=True):
        # Preprocess POS_CASH_balance.csv
        pos = pd.read_csv(
            '{}/POS_CASH_balance.csv'.format(self.data_path), nrows=num_rows)
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
        del pos
        gc.collect()
        return pos_agg

    def installments_payments(self, num_rows=None, nan_as_category=True):
        # Preprocess installments_payments.csv
        ins = pd.read_csv(
            '{}/installments_payments.csv'.format(self.data_path), nrows=num_rows)
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
        del ins
        gc.collect()
        return ins_agg

    def credit_card_balance(self, num_rows=None, nan_as_category=True):
        # Preprocess credit_card_balance.csv
        cc = pd.read_csv(
            '{}/credit_card_balance.csv'.format(self.data_path), nrows=num_rows)
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                                   for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg


class KaggleDataset_v5(KaggleDataset):
    '''Dataset class to load and process competition data'''

    def __init__(self, data_path, debug=False):
        self.data_path = data_path
        self.debug = debug

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

        # Feature Engineering
        df['annuity_income_percentage'] = df[
            'AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
        df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['credit_to_income_ratio'] = df[
            'AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['income_credit_percentage'] = df[
            'AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['income_per_child'] = df[
            'AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
        df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['phone_to_birth_ratio'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['phone_to_employ_ratio'] = df[
            'DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
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
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        df['short_employment'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)
        df['young_age'] = (df['DAYS_BIRTH'] < -14000).astype(int)

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

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
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
        return bureau_agg

    def previous_applications(self, num_rows=None, nan_as_category=True):
        # Preprocess previous_applications.csv
        prev = pd.read_csv(
            '{}/previous_application.csv'.format(self.data_path), nrows=num_rows)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
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
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg

    def pos_cash(self, num_rows=None, nan_as_category=True):
        # Preprocess POS_CASH_balance.csv
        pos = pd.read_csv(
            '{}/POS_CASH_balance.csv'.format(self.data_path), nrows=num_rows)
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
        del pos
        gc.collect()
        return pos_agg

    def installments_payments(self, num_rows=None, nan_as_category=True):
        # Preprocess installments_payments.csv
        ins = pd.read_csv(
            '{}/installments_payments.csv'.format(self.data_path), nrows=num_rows)
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
        del ins
        gc.collect()
        return ins_agg

    def credit_card_balance(self, num_rows=None, nan_as_category=True):
        # Preprocess credit_card_balance.csv
        cc = pd.read_csv(
            '{}/credit_card_balance.csv'.format(self.data_path), nrows=num_rows)

        cc['AMT_DRAWINGS_ATM_CURRENT'][
            cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
        cc['AMT_DRAWINGS_CURRENT'][cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # General aggregations
        static_features = self._static_features(cc)
        dynamic_features = self._dynamic_features(cc)

        cc_agg = pd.merge(static_features,
                          dynamic_features,
                          on=['SK_ID_CURR'],
                          validate='one_to_one')
        del cc
        gc.collect()
        return cc_agg

    def _static_features(self, credit_card, **kwargs):
        credit_card['number_of_installments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']

        credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

        features = pd.DataFrame(
            {'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        groupby = credit_card.groupby(by=['SK_ID_CURR'])

        g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={
                 'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['SK_DPD'].agg('mean').reset_index()
        g.rename(index=str, columns={
                 'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={
                 'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={
                 'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['number_of_installments'].agg('sum').reset_index()
        g.rename(index=str, columns={
                 'number_of_installments': 'credit_card_total_installments'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['credit_card_max_loading_of_credit_limit'].agg(
            'mean').reset_index()
        g.rename(index=str,
                 columns={
                     'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features[
            'credit_card_drawings_total']

        features['credit_card_installments_per_loan'] = (
            features['credit_card_total_installments'] / features['credit_card_number_of_loans'])

        return features

    def _dynamic_features(self, credit_card, **kwargs):
        features = pd.DataFrame(
            {'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        credit_card_sorted = credit_card.sort_values(
            ['SK_ID_CURR', 'MONTHS_BALANCE'])

        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])
        credit_card_sorted['credit_card_monthly_diff'] = groupby[
            'AMT_BALANCE'].diff()
        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])

        g = groupby['credit_card_monthly_diff'].agg('mean').reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        return features
