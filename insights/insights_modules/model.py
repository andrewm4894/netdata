import logging
import warnings

from insights_modules.model_adtk import do_adtk, adtk_models_supported, adtk_models_chart_level, adtk_meta_models
from insights_modules.model_ks import do_ks
from insights_modules.model_mp import do_mp, mp_models_supported
from insights_modules.model_pyod import do_pyod, pyod_models_supported

# filter some future warnings from sklearn and numba that come via pyod
#warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

models_supported = ['ks'] + mp_models_supported + pyod_models_supported + adtk_models_supported
models_chart_enabled = pyod_models_supported + adtk_models_chart_level
models_chart_only = ['iforest']


def run_model(model, colnames, arr_baseline, arr_highlight, n_lags=0, model_errors='ignore', model_level='dim'):
    model, model_level, n_lags = validate_inputs(model, model_level, n_lags)
    if model in pyod_models_supported:
        results = do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    elif model in mp_models_supported:
        results = do_mp(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    elif model in adtk_models_supported:
        results = do_adtk(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    elif model == 'ks':
        results = do_ks(colnames, arr_baseline, arr_highlight)
    else:
        raise ValueError(f"unknown model '{model}'")
    return results


def validate_inputs(model, model_level, n_lags):
    if model not in models_chart_enabled and model_level == 'chart':
        model_level = 'dim'
    if model in adtk_meta_models and n_lags == 0:
        n_lags = 1
    if model == 'ks':
        n_lags = 0
    if model in models_chart_only:
        model_level = 'chart'
    return model, model_level, n_lags


