import logging
import warnings

from insights_modules.model_adtk import do_adtk, adtk_models_supported
from insights_modules.model_ks import do_ks
from insights_modules.model_mp import do_mp, mp_models_supported
from insights_modules.model_pyod import do_pyod, pyod_models_supported

# filter some future warnings from sklearn and numba that come via pyod
#warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)


def run_model(model, colnames, arr_baseline, arr_highlight, n_lags=0, model_errors='ignore', model_level='dim'):
    if model in pyod_models_supported:
        results, summary = do_pyod(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    elif model in mp_models_supported:
        results, summary = do_mp(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    elif model in adtk_models_supported:
        results, summary = do_adtk(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors, model_level)
    else:
        results, summary = do_ks(colnames, arr_baseline, arr_highlight)
    return results, summary

