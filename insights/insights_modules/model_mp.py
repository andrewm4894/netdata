import logging

import numpy as np
import stumpy

from model_utils import init_counters, summary_info

log = logging.getLogger(__name__)

mp_models_supported = [
    'mp', 'mp_approx'
]


def do_mp(model, colnames, arr_baseline, arr_highlight):

    # init some counters
    n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail = init_counters(colnames)

    # dict to collect results into
    results = {}

    arr = np.concatenate((arr_baseline, arr_highlight))
    n_highlight = arr_highlight.shape[0]

    # loop over each col and build mp model
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

        fit_success += 1
        mp_highlight = mp[-n_highlight:]
        mp_thold = np.percentile(mp, 90)

        score = np.mean(np.where(mp_highlight >= mp_thold, 1, 0))
        if chart in results:
            results[chart].append({dimension: {'score': score}})
        else:
            results[chart] = [{dimension: {'score': score}}]

    # log some summary stats
    log.info(summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default))

    return results

