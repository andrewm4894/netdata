import logging

import numpy as np
from scipy.stats import ks_2samp
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD

log = logging.getLogger(__name__)

supported_pyod_models = [
    'abod', 'auto_encoder', 'cblof', 'hbos', 'iforest', 'knn', 'lmdd', 'loci', 'loda', 'lof', 'mcd', 'ocsvm',
    'pca', 'sod', 'vae', 'xgbod'
]


def do_ks(colnames, arr_baseline, arr_highlight):
    # dict to collect results into
    results = {}
    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        score, _ = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]
    return results


def do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags):
    # dict to collect results into
    results = {}
    # initial model set up
    if model == 'knn':
        clf = KNN(**model['params'])
    elif model == 'abod':
        clf = ABOD(**model['params'])
    elif model == 'auto_encoder':
        clf = AutoEncoder(**model['params'])
    elif model == 'cblof':
        clf = CBLOF(**model['params'])
    elif model == 'hbos':
        clf = HBOS(**model['params'])
    elif model == 'iforest':
        clf = IForest(**model['params'])
    elif model == 'lmdd':
        clf = LMDD(**model['params'])
    elif model == 'loci':
        clf = LOCI(**model['params'])
    elif model == 'loda':
        clf = LODA(**model['params'])
    elif model == 'lof':
        clf = LOF(**model['params'])
    elif model == 'mcd':
        clf = MCD(**model['params'])
    elif model == 'ocsvm':
        clf = OCSVM(**model['params'])
    elif model == 'pca':
        clf = PCA(**model['params'])
    elif model == 'sod':
        clf = SOD(**model['params'])
    elif model == 'vae':
        clf = VAE(**model['params'])
    elif model == 'xgbod':
        clf = XGBOD(**model['params'])
    else:
        raise ValueError(f"unknown model {model}")
    # fit model for each dimension and then use model to score highlighted area
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):
        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        arr_baseline_dim = arr_baseline[:, [n]]
        arr_highlight_dim = arr_highlight[:, [n]]
        if n_lags > 0:
            arr_baseline_dim = add_lags(arr_baseline_dim, n_lags=n_lags)
            arr_highlight_dim = add_lags(arr_highlight_dim, n_lags=n_lags)
        # remove any nan rows
        arr_baseline_dim = arr_baseline_dim[~np.isnan(arr_baseline_dim).any(axis=1)]
        arr_highlight_dim = arr_highlight_dim[~np.isnan(arr_highlight_dim).any(axis=1)]
        #log.info(f'... chart = {chart}')
        #log.info(f'... dimension = {dimension}')
        #log.info(f'... arr_baseline_dim.shape = {arr_baseline_dim.shape}')
        #log.info(f'... arr_highlight_dim.shape = {arr_highlight_dim.shape}')
        #log.info(f'... arr_baseline_dim = {arr_baseline_dim}')
        #log.info(f'... arr_highlight_dim = {arr_highlight_dim}')
        # try fit and if fails fallback to default model
        clf.fit(arr_baseline_dim)
        #try:
        #    clf.fit(arr_baseline_dim)
        #except:
        #    clf = DefaultPyODModel()
        #    clf.fit(arr_baseline_dim)
        # 0/1 anomaly predictions
        preds = clf.predict(arr_highlight_dim)
        #log.info(f'... preds.shape = {preds.shape}')
        #log.info(f'... preds = {preds}')
        # anomaly probability scores
        probs = clf.predict_proba(arr_highlight_dim)[:, 1]
        #log.info(f'... probs.shape = {probs.shape}')
        #log.info(f'... probs = {probs}')
        # save results
        score = (np.mean(probs) + np.mean(preds))/2
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]
    return results


def run_model(model, colnames, arr_baseline, arr_highlight, n_lags):
    if model in supported_pyod_models:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags)
    else:
        results = do_ks(colnames, arr_baseline, arr_highlight)
    return results

