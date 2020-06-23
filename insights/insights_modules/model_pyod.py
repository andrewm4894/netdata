import logging

import numpy as np
from pyod.models.hbos import HBOS as PyODDefaultModel

from model_utils import init_counters, try_fit, summary_info, get_col_map
from utils import add_lags

log = logging.getLogger(__name__)

pyod_models_supported = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]


def do_pyod(model, colnames, arr_baseline, arr_highlight):

    # init some counters
    n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail = init_counters(colnames)

    # dict to collect results into
    results = {}

    n_lags = model.get('n_lags', 0)
    model_level = model.get('model_level', 'dim')
    model = model.get('type', 'hbos')

    # model init
    clf = pyod_init(model)

    # get map of cols to loop over
    col_map = get_col_map(colnames, model_level)

    # build each model
    for colname in col_map:

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1] if '|' in colname else '*'
        arr_baseline_dim = arr_baseline[:, col_map[colname]]
        arr_highlight_dim = arr_highlight[:, col_map[colname]]

        # check for bad data
        bad_data = False

        # skip if bad data
        if bad_data:

            n_bad_data += 1
            log.info(f'... skipping {colname} due to bad data')

        else:

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

            if model == ['auto_encoder']:
                clf = pyod_init(model, n_features=arr_baseline_dim.shape[1])

            clf, result = try_fit(clf, colname, arr_baseline_dim, PyODDefaultModel)
            fit_success += 1 if result == 'success' else 0
            fit_default += 1 if result == 'default' else 0

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
            if chart in results:
                results[chart].append({dimension: {'score': score}})
            else:
                results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level))

    return results


def pyod_init(model, n_features=None):
    # initial model set up
    if model == 'abod':
        from pyod.models.abod import ABOD
        clf = ABOD()
    elif model == 'auto_encoder' and n_features:
        #import os
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from pyod.models.auto_encoder import AutoEncoder
        clf = AutoEncoder(
            hidden_neurons=[n_features, n_features*5, n_features*5, n_features],
            epochs=5, batch_size=64, preprocessing=False
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
        #raise ValueError(f"unknown model {model}")
        clf = PyODDefaultModel()
    return clf

