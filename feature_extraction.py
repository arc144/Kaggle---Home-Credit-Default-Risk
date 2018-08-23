from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, iqr, skew
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import gc
from sklearn.externals import joblib


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in tqdm(enumerate(groupby_object), total=n_groups):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[
                feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[
                feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[
                feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[
                feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[
                feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[
                feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(
                gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(
                gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(
                gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[
                feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


class BasicHandCraftedFeatures():

    def __init__(self, num_workers=1, **kwargs):
        self.num_workers = num_workers
        self.features = None

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def transform(self, **kwargs):
        return {'features_table': self.features}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)


class POSCASHBalanceFeatures(BasicHandCraftedFeatures):

    def __init__(self, last_k_agg_periods, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.num_workers = num_workers
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_trend_periods = last_k_trend_periods

    def fit(self, pos_cash, **kwargs):
        pos_cash['is_contract_status_completed'] = pos_cash[
            'NAME_CONTRACT_STATUS'] == 'Completed'
        pos_cash['pos_cash_paid_late'] = (pos_cash['SK_DPD'] > 0).astype(int)
        pos_cash['pos_cash_paid_late_with_tolerance'] = (
            pos_cash['SK_DPD_DEF'] > 0).astype(int)

        features = pd.DataFrame(
            {'SK_ID_CURR': pos_cash['SK_ID_CURR'].unique()})
        groupby = pos_cash.groupby(['SK_ID_CURR'])
        func = partial(POSCASHBalanceFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                           num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        one_time = POSCASHBalanceFeatures.one_time_features(gr)
        gc.collect()
        all = POSCASHBalanceFeatures.all_installment_features(gr)
        gc.collect()
        agg = POSCASHBalanceFeatures.last_k_installment_features(
            gr, agg_periods)
        gc.collect()
        trend = POSCASHBalanceFeatures.trend_in_last_k_installment_features(
            gr, trend_periods)
        gc.collect()
        last = POSCASHBalanceFeatures.last_loan_features(gr)
        gc.collect()
        features = {**one_time, **all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def one_time_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
        features = {}

        features['pos_cash_remaining_installments'] = gr_[
            'CNT_INSTALMENT_FUTURE'].tail(1)
        features['pos_cash_completed_contracts'] = gr_[
            'is_contract_status_completed'].agg('sum')

        return features

    @staticmethod
    def all_installment_features(gr):
        return POSCASHBalanceFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['sum', 'mean', 'max',
                                                 'std', 'skew', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['sum', 'mean', 'max',
                                                 'std', 'skew', 'kurt'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'SK_DPD', '{}_period_trend_'.format(
                                             period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'SK_DPD_DEF', '{}_period_trend_'.format(
                                             period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(
                                             period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['count', 'sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')

        return features


class PreviousApplicationFeatures(BasicHandCraftedFeatures):

    def __init__(self, numbers_of_applications=[], num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.numbers_of_applications = numbers_of_applications

    def fit(self, prev_applications, **kwargs):
        features = pd.DataFrame(
            {'SK_ID_CURR': prev_applications['SK_ID_CURR'].unique()})

        prev_app_sorted = prev_applications.sort_values(
            ['SK_ID_CURR', 'DAYS_DECISION'])
        prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

        prev_app_sorted['previous_application_prev_was_approved'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
        g = prev_app_sorted_groupby[
            'previous_application_prev_was_approved'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['previous_application_prev_was_refused'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        g = prev_app_sorted_groupby[
            'previous_application_prev_was_refused'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={
                 'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])[
            'previous_application_prev_was_refused'].mean().reset_index()
        g.rename(index=str, columns={
            'previous_application_prev_was_refused': 'previous_application_fraction_of_refused_applications'},
            inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['prev_applications_prev_was_revolving_loan'] = (
            prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])[
            'prev_applications_prev_was_revolving_loan'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        for number in self.numbers_of_applications:
            prev_applications_tail = prev_app_sorted_groupby.tail(number)

            tail_groupby = prev_applications_tail.groupby(by=['SK_ID_CURR'])

            g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(
                         number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(
                             number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

        self.features = features
        return self


class CreditCardBalanceFeatures(BasicHandCraftedFeatures):

    def fit(self, credit_card, **kwargs):
        static_features = self._static_features(credit_card, **kwargs)
        dynamic_features = self._dynamic_features(credit_card, **kwargs)

        self.features = pd.merge(static_features,
                                 dynamic_features,
                                 on=['SK_ID_CURR'],
                                 validate='one_to_one')
        return self

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


class InstallmentPaymentsFeatures(BasicHandCraftedFeatures):

    def __init__(self, last_k_agg_periods, last_k_agg_period_fractions, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions
        self.last_k_trend_periods = last_k_trend_periods

        self.num_workers = num_workers
        self.features = None

    def fit(self, installments, **kwargs):
        installments['installment_paid_late_in_days'] = installments['DAYS_ENTRY_PAYMENT'] - installments[
            'DAYS_INSTALMENT']
        installments['installment_paid_late'] = (
            installments['installment_paid_late_in_days'] > 0).astype(int)
        installments['installment_paid_over_amount'] = installments[
            'AMT_PAYMENT'] - installments['AMT_INSTALMENT']
        installments['installment_paid_over'] = (
            installments['installment_paid_over_amount'] > 0).astype(int)

        features = pd.DataFrame(
            {'SK_ID_CURR': installments['SK_ID_CURR'].unique()})
        groupby = installments.groupby(['SK_ID_CURR'])

        func = partial(InstallmentPaymentsFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       period_fractions=self.last_k_agg_period_fractions,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                           num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods, period_fractions):
        all = InstallmentPaymentsFeatures.all_installment_features(gr)
        agg = InstallmentPaymentsFeatures.last_k_installment_features_with_fractions(gr,
                                                                                     agg_periods,
                                                                                     period_fractions)
        trend = InstallmentPaymentsFeatures.trend_in_last_k_installment_features(
            gr, trend_periods)
        last = InstallmentPaymentsFeatures.last_loan_features(gr)
        features = {**all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return InstallmentPaymentsFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features_with_fractions(gr, periods, period_fractions):
        features = InstallmentPaymentsFeatures.last_k_installment_features(
            gr, periods)

        for short_period, long_period in period_fractions:
            short_feature_names = get_feature_names_by_period(
                features, short_period)
            long_feature_names = get_feature_names_by_period(
                features, long_period)

            for short_feature, long_feature in zip(short_feature_names, long_feature_names):
                old_name_chunk = '_{}_'.format(short_period)
                new_name_chunk = '_{}by{}_fraction_'.format(
                    short_period, long_period)
                fraction_feature_name = short_feature.replace(
                    old_name_chunk, new_name_chunk)
                features[fraction_feature_name] = safe_div(
                    features[short_feature], features[long_feature])
        return features

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                             ['sum', 'mean', 'max', 'min', 'std',
                                                 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)

            features = add_features_in_group(features, gr_period, 'installment_paid_late_in_days',
                                             ['sum', 'mean', 'max', 'min', 'std',
                                                 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over_amount',
                                             ['sum', 'mean', 'max', 'min', 'std',
                                                 'median', 'skew', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over',
                                             ['count', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'installment_paid_late_in_days', '{}_period_trend_'.format(
                                             period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'installment_paid_over_amount', '{}_period_trend_'.format(
                                             period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late',
                                         ['count', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over',
                                         ['count', 'mean'],
                                         'last_loan_')
        return features


class GroupbyMerge():

    def __init__(self, id_columns, **kwargs):
        # super().__init__()
        self.id_columns = id_columns

    def _feature_names(self, features):
        feature_names = list(features.columns)
        feature_names.remove(self.id_columns[0])
        return feature_names

    def transform(self, table, features, **kwargs):
        table = table.merge(features,
                            left_on=[self.id_columns[0]],
                            right_on=[self.id_columns[1]],
                            how='left',
                            validate='one_to_one')
        return table.astype(np.float32)

if __name__ == '__main__':
    DATA_PATH = './data/'
    df = pd.read_csv(
        '{}/application_train.csv'.format(DATA_PATH), nrows=1000)
    poscsv = pd.read_csv(
        '{}/POS_CASH_balance.csv'.format(DATA_PATH), nrows=1000)
    pos = POSCASHBalanceFeatures([1, 5, 10, 20, 50, 100], [10, 50, 100, 500])
    pos = pos.fit(poscsv)
    gm = GroupbyMerge(id_columns=('SK_ID_CURR', 'SK_ID_CURR'))
    features = gm.transform(df, pos.features)
