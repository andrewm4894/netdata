import logging
import warnings

import numpy as np
from scipy.stats import ks_2samp
from pyod.models.hbos import HBOS as DefaulyPyODModel
import stumpy

# filter some future warnings from sklearn that come via pyod
warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

supported_pyod_models = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]


def run_model(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors='ignore'):

    if model in supported_pyod_models:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors)
    elif model in ['mp', 'mp_approx']:
        results = do_mp(colnames, arr_baseline, arr_highlight, model=model)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)

    return results


def do_ks(colnames, arr_baseline, arr_highlight):

    # dict to collect results into
    results = {}

    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):

        # extract chart and dim from colname
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]

        # just get dim of interest
        arr_baseline_dim = arr_baseline[:, n]
        arr_highlight_dim = arr_highlight[:, n]

        log.debug(f'... chart = {chart}')
        log.debug(f'... dimension = {dimension}')
        log.debug(f'... arr_baseline_dim.shape = {arr_baseline_dim.shape}')
        log.debug(f'... arr_highlight_dim.shape = {arr_highlight_dim.shape}')
        log.debug(f'... arr_baseline_dim = {arr_baseline_dim}')
        log.debug(f'... arr_highlight_dim = {arr_highlight_dim}')

        # save results
        score, _ = ks_2samp(arr_baseline_dim, arr_highlight_dim, mode='asymp')
        results = save_results(results, chart, dimension, score)

    return results


def do_mp(colnames, arr_baseline, arr_highlight, model='mp'):
    arr = np.concatenate((arr_baseline, arr_highlight))
    n_baseline = arr_baseline.shape[0]
    n_highlight = arr_highlight.shape[0]
    # dict to collect results into
    results = {}
    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        m = 30
        if model == 'mp':
            mp = stumpy.stump(arr[:, n], m)[:, 0]
        elif model == 'mp_approx':
            approx = stumpy.scrump(arr[:, n], m, percentage=0.01, pre_scrump=True)
            for _ in range(20):
                approx.update()
            mp = approx.P_
        else:
            raise ValueError(f"... unknown model '{model}'")
        mp_highlight = mp[0:n_highlight]
        mp_thold = np.percentile(mp, 90)
        score = np.mean(np.where(mp_highlight >= mp_thold, 1, 0))
        results = save_results(results, chart, dimension, score)
    return results


def do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors='default'):

    # dict to collect results into
    results = {}

    # initialise a pyod model
    n_train, n_features = arr_baseline.shape[0], 1+n_lags
    clf = pyod_init(model, n_train, n_features)

    # fit model for each dimension and then use model to score highlighted area
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):

        # extract chart and dim from colname
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]

        # just get dim of interest
        arr_baseline_dim = arr_baseline[:, [n]]
        arr_highlight_dim = arr_highlight[:, [n]]

        # add lags if needed
        if n_lags > 0:
            arr_baseline_dim = add_lags(arr_baseline_dim, n_lags=n_lags)
            arr_highlight_dim = add_lags(arr_highlight_dim, n_lags=n_lags)

        # remove any nan rows
        arr_baseline_dim = arr_baseline_dim[~np.isnan(arr_baseline_dim).any(axis=1)]
        arr_highlight_dim = arr_highlight_dim[~np.isnan(arr_highlight_dim).any(axis=1)]

        log.debug(f'... chart = {chart}')
        log.debug(f'... dimension = {dimension}')
        log.debug(f'... arr_baseline_dim.shape = {arr_baseline_dim.shape}')
        log.debug(f'... arr_highlight_dim.shape = {arr_highlight_dim.shape}')
        log.debug(f'... arr_baseline_dim = {arr_baseline_dim}')
        log.debug(f'... arr_highlight_dim = {arr_highlight_dim}')

        # fit model
        try:
            clf.fit(arr_baseline_dim)
        except Exception as e:
            if model_errors == 'default':
                log.warning(f"... warning could not fit model for {colname}, trying default")
                clf = DefaulyPyODModel()
                clf.fit(arr_baseline_dim)
            elif model_errors == 'ignore':
                log.warning(f"... warning could not fit model for {colname}, skipping")
                continue
            else:
                log.error('hello')
                raise e

        # 0/1 anomaly predictions
        preds = clf.predict(arr_highlight_dim)

        log.debug(f'... preds.shape = {preds.shape}')
        log.debug(f'... preds = {preds}')

        # anomaly probability scores
        probs = clf.predict_proba(arr_highlight_dim)[:, 1]

        log.debug(f'... probs.shape = {probs.shape}')
        log.debug(f'... probs = {probs}')

        # save results
        score = (np.mean(probs) + np.mean(preds))/2
        results = save_results(results, chart, dimension, score)

    return results


def save_results(results, chart, dimension, score):
    if chart in results:
        results[chart].update({dimension: {"score": round(score, 4)}})
    else:
        results[chart] = {dimension: {"score": round(score, 4)}}
    return results


def add_lags(arr, n_lags=1):
    arr_orig = np.copy(arr)
    for n_lag in range(1, n_lags + 1):
        arr = np.concatenate((arr, np.roll(arr_orig, n_lag, axis=0)), axis=1)
    arr = arr[n_lags:]
    log.debug(f'... n_lags = {n_lags} arr_orig.shape = {arr_orig.shape}  arr.shape = {arr.shape}')
    return arr


def pyod_init(model, n_train=None, n_features=None):
    # initial model set up
    if model == 'abod':
        from pyod.models.abod import ABOD
        clf = ABOD()
    elif model == 'auto_encoder':
        import os
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(
            hidden_neurons=[n_features, n_features*5, n_features*5, n_features], epochs=5,
            batch_size=64, preprocessing=False
        )
    elif model == 'cblof':
        from pyod.models.cblof import CBLOF
        clf = CBLOF(n_clusters=4)
    elif model == 'hbos':
        from pyod.models.hbos import HBOS
        clf = HBOS()
    elif model == 'iforest':
        from pyod.models.iforest import IForest
        clf = IForest()
    elif model == 'knn':
        from pyod.models.knn import KNN
        clf = KNN()
    elif model == 'lmdd':
        from pyod.models.lmdd import LMDD
        clf = LMDD()
    elif model == 'loci':
        from pyod.models.loci import LOCI
        clf = LOCI()
    elif model == 'loda':
        from pyod.models.loda import LODA
        clf = LODA()
    elif model == 'lof':
        from pyod.models.lof import LOF
        clf = LOF()
    elif model == 'mcd':
        from pyod.models.mcd import MCD
        clf = MCD()
    elif model == 'ocsvm':
        from pyod.models.ocsvm import OCSVM
        clf = OCSVM()
    elif model == 'pca':
        from pyod.models.pca import PCA
        clf = PCA()
    elif model == 'sod':
        from pyod.models.sod import SOD
        clf = SOD()
    elif model == 'vae':
        from pyod.models.vae import VAE
        clf = VAE()
    elif model == 'xgbod':
        from pyod.models.xgbod import XGBOD
        clf = XGBOD()
    else:
        raise ValueError(f"unknown model {model}")
    return clf

