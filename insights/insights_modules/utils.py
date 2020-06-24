import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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



