import logging

import numpy as np

log = logging.getLogger(__name__)


def normalize_results(results):

    # get max and min scores
    scores = []
    for chart in results:
        for dimension in results[chart]:
            scores.append(results[chart][dimension]['score'])
    score_max = max(scores)
    score_min = min(scores)

    # normalize scores
    for chart in results:
        for dimension in results[chart]:
            score = results[chart][dimension]['score']
            score_norm = round((score - score_min) / (score_max - score_min), 4)
            results[chart][dimension]['score_norm'] = score_norm

    return results


def add_lags(data, n_lags=1, data_type='np'):
    data_orig = np.copy(data)
    if data_type == 'np':
        for n_lag in range(1, n_lags + 1):
            data = np.concatenate((data, np.roll(data_orig, n_lag, axis=0)), axis=1)
        data = data[n_lags:]
    elif data_type == 'df':
        colnames_new = [f"{col}_lag{n_lag}".replace('_lag0', '') for n_lag in range(n_lags+1) for col in data.columns]
        data = pd.concat([data.shift(n_lag) for n_lag in range(n_lags + 1)], axis=1)
        data.columns = colnames_new
    log.debug(f'... (add_lags) n_lags = {n_lags} arr_orig.shape = {data_orig.shape}  arr.shape = {data.shape}')
    return data



