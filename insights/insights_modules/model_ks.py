import logging

from scipy.stats import ks_2samp

from insights_modules.model_utils import init_counters, save_results, summary_dict

log = logging.getLogger(__name__)


def do_ks(colnames, arr_baseline, arr_highlight, model_level='dim', n_lags=0):

    # init some counters
    n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail = init_counters(colnames)

    # dict to collect results into
    results = {}

    # loop over each col and do the ks test
    for colname, n in zip(colnames, range(arr_baseline.shape[1])):

        chart = colname.split('|')[0]
        dimension = colname.split('|')[1]
        score, _ = ks_2samp(arr_baseline[:, n], arr_highlight[:, n], mode='asymp')
        fit_success += 1
        results = save_results(results, chart, dimension, score)

    # add summary to results
    results['summary'] = summary_dict(
        n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level, n_lags
    )

    return results

