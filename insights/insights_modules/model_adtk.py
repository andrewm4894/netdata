import logging

import pandas as pd
from adtk.detector import InterQuartileRangeAD as ADTKDefaultModel

from model_utils import init_counters, try_fit, summary_info, get_col_map
from utils import add_lags

log = logging.getLogger(__name__)

adtk_models_supported = [
    'iqr', 'ar', 'esd', 'level', 'persist', 'quantile', 'seasonal', 'volatility', 'kmeans', 'birch', 'eliptic',
    'pcaad', 'linear', 'gmm', 'vbgmm', 'isof', 'lofad', 'mcdad', 'rf', 'huber', 'knnad', 'kernridge'
]
adtk_models_lags_allowed = [
    'kmeans', 'birch', 'gmm', 'eliptic', 'vbgmm', 'isof', 'lofad', 'mcdad', 'linear', 'rf', 'huber', 'knnad',
    'kernridge'
]
adtk_models_chart_level = [
    'kmeans', 'birch', 'gmm', 'eliptic', 'vbgmm', 'isof', 'lofad', 'mcdad', 'linear', 'rf', 'huber', 'knnad',
    'kernridge'
]
adtk_meta_models = ['linear', 'rf', 'huber', 'knnad', 'kernridge']


def do_adtk(model, colnames, arr_baseline, arr_highlight):

    # init some counters
    n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail = init_counters(colnames)

    # dict to collect results into
    results = {}

    n_lags = model.get('n_lags', 0)
    model_level = model.get('model_level', 'dim')
    model = model.get('type', 'iqr')

    df_baseline = pd.DataFrame(arr_baseline, columns=colnames)
    df_baseline = df_baseline.set_index(pd.DatetimeIndex(pd.to_datetime(df_baseline.index, unit='s'), freq='1s'))
    df_highlight = pd.DataFrame(arr_highlight, columns=colnames)
    df_highlight = df_highlight.set_index(pd.DatetimeIndex(pd.to_datetime(df_highlight.index, unit='s'), freq='1s'))

    # model init
    clf = adtk_init(model)

    # get map of cols to loop over
    col_map = get_col_map(colnames, model_level)

    # build each model
    for colname in col_map:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1] if '|' in colname else '*'

        log.debug(f'... chart = {chart}')
        log.debug(f'... dimension = {dimension}')

        df_baseline_dim = df_baseline.iloc[:, col_map[colname]]
        df_highlight_dim = df_highlight.iloc[:, col_map[colname]]

        # check for bad data
        bad_data = False
        baseline_dim_na_pct = max(df_baseline_dim.isna().sum() / len(df_baseline))
        highlight_dim_na_pct = max(df_highlight_dim.isna().sum() / len(df_highlight))
        if baseline_dim_na_pct >= 0.1:
            bad_data = True
        if highlight_dim_na_pct >= 0.1:
            bad_data = True

        # skip if bad data
        if bad_data:

            n_bad_data += 1
            log.info(f'... skipping {colname} due to bad data')

        else:

            if model in adtk_models_lags_allowed:

                if n_lags > 0:

                    df_baseline_dim = add_lags(df_baseline_dim, n_lags, 'df')
                    df_highlight_dim = add_lags(df_highlight_dim, n_lags, 'df')

            df_baseline_dim = df_baseline_dim.dropna()
            df_highlight_dim = df_highlight_dim.dropna()

            log.debug(f'... chart = {chart}')
            log.debug(f'... dimension = {dimension}')
            log.debug(f'... df_baseline_dim.shape = {df_baseline_dim.shape}')
            log.debug(f'... df_highlight_dim.shape = {df_highlight_dim.shape}')
            log.debug(f'... df_baseline_dim = {df_baseline_dim}')
            log.debug(f'... df_highlight_dim = {df_highlight_dim}')

            # reinit model if needed
            if model in adtk_meta_models:
                clf = adtk_init(model, colname)

            clf, result = try_fit(clf, colname, df_baseline_dim, ADTKDefaultModel)
            fit_success += 1 if result == 'success' else 0
            fit_default += 1 if result == 'default' else 0

            # get scores
            preds = clf.predict(df_highlight_dim)

            log.debug(f'... preds.shape = {preds.shape}')
            log.debug(f'... preds = {preds}')

            score = preds.mean().mean()
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level))

    return results


def adtk_init(model, colname=None):
    if model == 'iqr':
        from adtk.detector import InterQuartileRangeAD
        clf = InterQuartileRangeAD()
    elif model == 'ar':
        from adtk.detector import AutoregressionAD
        clf = AutoregressionAD()
    elif model == 'esd':
        from adtk.detector import GeneralizedESDTestAD
        clf = GeneralizedESDTestAD()
    elif model == 'level':
        from adtk.detector import LevelShiftAD
        clf = LevelShiftAD(15)
    elif model == 'persist':
        from adtk.detector import PersistAD
        clf = PersistAD(15)
    elif model == 'quantile':
        from adtk.detector import QuantileAD
        clf = QuantileAD()
    elif model == 'seasonal':
        from adtk.detector import SeasonalAD
        clf = SeasonalAD()
    elif model == 'volatility':
        from adtk.detector import VolatilityShiftAD
        clf = VolatilityShiftAD(15)
    elif model == 'kmeans':
        from adtk.detector import MinClusterDetector
        from sklearn.cluster import KMeans
        clf = MinClusterDetector(KMeans(n_clusters=2))
    elif model == 'birch':
        from adtk.detector import MinClusterDetector
        from sklearn.cluster import Birch
        clf = MinClusterDetector(Birch(threshold=0.25, branching_factor=25))
    elif model == 'gmm':
        from adtk.detector import MinClusterDetector
        from sklearn.mixture import GaussianMixture
        clf = MinClusterDetector(GaussianMixture(n_components=2, max_iter=50))
    elif model == 'vbgmm':
        from adtk.detector import MinClusterDetector
        from sklearn.mixture import BayesianGaussianMixture
        clf = MinClusterDetector(BayesianGaussianMixture(n_components=2, max_iter=50))
    elif model == 'eliptic':
        from adtk.detector import OutlierDetector
        from sklearn.covariance import EllipticEnvelope
        clf = OutlierDetector(EllipticEnvelope())
    elif model == 'mcdad':
        from adtk.detector import OutlierDetector
        from sklearn.covariance import MinCovDet
        clf = OutlierDetector(MinCovDet())
    elif model == 'isof':
        from adtk.detector import OutlierDetector
        from sklearn.ensemble import IsolationForest
        clf = OutlierDetector(IsolationForest())
    elif model == 'lofad':
        from adtk.detector import OutlierDetector
        from sklearn.neighbors import LocalOutlierFactor
        clf = OutlierDetector(LocalOutlierFactor())
    elif model == 'pcaad':
        from adtk.detector import PcaAD
        clf = PcaAD()
    elif model == 'linear':
        from adtk.detector import RegressionAD
        from sklearn.linear_model import LinearRegression
        clf = RegressionAD(LinearRegression(), target=colname)
    elif model == 'rf':
        from adtk.detector import RegressionAD
        from sklearn.ensemble import RandomForestRegressor
        clf = RegressionAD(RandomForestRegressor(), target=colname)
    elif model == 'huber':
        from adtk.detector import RegressionAD
        from sklearn.linear_model import HuberRegressor
        clf = RegressionAD(HuberRegressor(), target=colname)
    elif model == 'knnad':
        from adtk.detector import RegressionAD
        from sklearn.neighbors import KNeighborsRegressor
        clf = RegressionAD(KNeighborsRegressor(), target=colname)
    elif model == 'kernridge':
        from adtk.detector import RegressionAD
        from sklearn.kernel_ridge import KernelRidge
        clf = RegressionAD(KernelRidge(), target=colname)
    else:
        clf = ADTKDefault()
    return clf