# -*- coding: utf-8 -*-
# Description: anomalies netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

import time
from datetime import datetime
import re

import requests
import numpy as np
import pandas as pd
from netdata_pandas.data import get_data, get_allmetrics
from pysad.models import xStream
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from sklearn.preprocessing import MinMaxScaler

from bases.FrameworkServices.SimpleService import SimpleService


priority = 50
update_every: 2

ORDER = ['score']

CHARTS = {
    'score': {
        'options': ['score', 'Anomaly Score', 'score', 'anomalies_online', 'anomalies_online.score', 'line'],
        'lines': []
    },
}


class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.host = self.configuration.get('host', '127.0.0.1:19999')
        self.charts_regex = re.compile(self.configuration.get('charts_regex','system\..*'))
        self.charts_in_scope = list(filter(self.charts_regex.match, [c for c in requests.get(f'http://{self.host}/api/v1/charts').json()['charts'].keys()]))
        self.model = self.configuration.get('model', 'xstream')
        #self.train_max_n = self.configuration.get('train_max_n', 100000)
        #self.train_n_secs = self.configuration.get('train_n_secs', 14400)
        #self.offset_n_secs = self.configuration.get('offset_n_secs', 0)
        #self.train_every_n = self.configuration.get('train_every_n', 900)
        #self.contamination = self.configuration.get('contamination', 0.001)
        self.lags_n = self.configuration.get('lags_n', 5)
        self.smooth_n = self.configuration.get('smooth_n', 3)
        self.diffs_n = self.configuration.get('diffs_n', 1)
        self.custom_models = self.configuration.get('custom_models', None)
        self.custom_models_normalize = bool(self.configuration.get('custom_models_normalize', False))
        if self.custom_models:
            self.custom_models_names = [m['name'] for m in self.custom_models]
            self.custom_models_dims = [i for s in [m['dimensions'].split(',') for m in self.custom_models] for i in s]
            self.custom_models_charts = list(set([c.split('|')[0] for c in self.custom_models_dims]))
            self.custom_models_dims_renamed = []
            for m in self.custom_models:
                self.custom_models_dims_renamed.extend([f"{m['name']}.{d}" for d in m['dimensions'].split(',')])
            self.models_in_scope = list(set(self.charts_in_scope + self.custom_models_names))
            self.charts_in_scope = list(set(self.charts_in_scope + self.custom_models_charts))
        else:
            self.models_in_scope = self.charts_in_scope
        if self.model == 'xstream':
            self.models = {model: xStream() for model in self.models_in_scope}
        else:
            self.models = {model: xStream() for model in self.models_in_scope}
        self.preprocessor = {model: InstanceUnitNormScaler() for model in self.models_in_scope}
        self.postprocessor = {model: RunningAveragePostprocessor(window_size=self.smooth_n) for model in self.models_in_scope}
        self.df_allmetrics = pd.DataFrame()
        self.expected_cols = []
        self.data_latest = {}
        self.scaler = MinMaxScaler()

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        """If dimension not in chart then add it.
        """
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    @staticmethod
    def get_array_cols(colnames, arr, starts_with):
        """Given an array and list of colnames, return subset of cols from array where colname startswith starts_with.

        :param colnames <list>: list of colnames corresponding to arr.
        :param arr <np.ndarray>: numpy array we want to select a subset of cols from.
        :param starts_with <str>: the string we want to return all columns that start with given str value.
        :return: <np.ndarray> subseted array.
        """
        cols_idx = [i for i, x in enumerate(colnames) if x.startswith(starts_with)]

        return arr[:, cols_idx]

    def add_custom_models_dims(self, df):
        """Given a df, select columns used by custom models, add custom model name as prefix, and append to df.

        :param df <pd.DataFrame>: dataframe to append new renamed columns to.
        :return: <pd.DataFrame> dataframe with additional columns added relating to the specified custom models.
        """
        df_custom = df[self.custom_models_dims].copy()
        df_custom.columns = self.custom_models_dims_renamed
        df = df.join(df_custom)

        return df

    def set_expected_cols(self, df):
        """Given a df, set expected columns that determine expected schema for both training and prediction.

        :param df <pd.DataFrame>: dataframe to use to determine expected cols from.
        """
        self.expected_cols = sorted(list(set(df.columns)))
        # if using custom models then may need to remove some unused cols as data comes in per chart
        if self.custom_models:
            ignore_cols = []
            for col in self.expected_cols:
                for chart in self.custom_models_charts:
                    if col.startswith(chart):
                        if col not in self.custom_models_dims:
                            ignore_cols.append(col)
            self.expected_cols = [c for c in self.expected_cols if c not in ignore_cols]

    def make_features(self, arr, colnames, train=False):
        """Take in numpy array and preprocess accordingly by taking diffs, smoothing and adding lags.

        :param arr <np.ndarray>: numpy array we want to make features from.
        :param colnames <list>: list of colnames corresponding to arr.
        :param train <bool>: True if making features for training, in which case need to fit_transform scaler and maybe sample train_max_n.
        :return: (<np.ndarray>, <list>) tuple of list of colnames of features and transformed numpy array.
        """

        def lag(arr, n):
            res = np.empty_like(arr)
            res[:n] = np.nan
            res[n:] = arr[:-n]

            return res

        arr = np.nan_to_num(arr)

        if self.custom_models_normalize:
            # normalize just custom model columns which will be the last len(self.custom_models_dims) cols in arr.
            if train:
                arr_custom = self.scaler.fit_transform(arr[:,-len(self.custom_models_dims):])
            else:
                arr_custom = self.scaler.transform(arr[:,-len(self.custom_models_dims):])
            arr = arr[:,:-len(self.custom_models_dims)]
            arr = np.concatenate((arr, arr_custom), axis=1)

        if self.diffs_n > 0:
            arr = np.diff(arr, self.diffs_n, axis=0)
            arr = arr[~np.isnan(arr).any(axis=1)]

        if self.smooth_n > 1:
            arr = np.cumsum(arr, axis=0, dtype=float)
            arr[self.smooth_n:] = arr[self.smooth_n:] - arr[:-self.smooth_n]
            arr = arr[self.smooth_n - 1:] / self.smooth_n
            arr = arr[~np.isnan(arr).any(axis=1)]

        if self.lags_n > 0:
            colnames = colnames + [f'{col}_lag{lag}' for lag in range(1, self.lags_n + 1) for col in colnames]
            arr_orig = np.copy(arr)
            for lag_n in range(1, self.lags_n + 1):
                arr = np.concatenate((arr, lag(arr_orig, lag_n)), axis=1)
            arr = arr[~np.isnan(arr).any(axis=1)]

        if train:
            if len(arr) > self.train_max_n:
                arr = arr[np.random.randint(arr.shape[0], size=self.train_max_n), :]

        arr = np.nan_to_num(arr)

        return arr, colnames

    def predict(self):
        """Get latest data, make it into a feature vector, and get predictions for each available model.

        :return: (<dict>,<dict>) tuple of dictionaries, one for probability scores and the other for anomaly predictions.
        """
        # get latest data to predict on
        df_allmetrics = get_allmetrics(self.host, self.charts_in_scope, wide=True, sort_cols=True)
        self.set_expected_cols(df_allmetrics)
        df_allmetrics = df_allmetrics[self.expected_cols]
        if self.custom_models:
            df_allmetrics = self.add_custom_models_dims(df_allmetrics)
        self.df_allmetrics = self.df_allmetrics.append(df_allmetrics).ffill().tail((self.lags_n + self.smooth_n + self.diffs_n) * 2)

        # make feature vector
        X, feature_colnames = self.make_features(self.df_allmetrics.values, list(df_allmetrics.columns))
        data = self.try_predict(X, feature_colnames)

        return data

    def try_predict(self, X, feature_colnames):
        """Try make prediction and fall back to last known prediction if fails.

        :param X <np.ndarray>: feature vector.
        :param feature_colnames <list>: list of corresponding feature names.
        :return: (<dict>,<dict>) tuple of dictionaries, one for probability scores and the other for anomaly predictions.
        """
        data = {}
        for model in self.models.keys():
            X_model = self.get_array_cols(feature_colnames, X, starts_with=model)
            try:
                X_model = self.preprocessor[model].fit_transform_partial(X_model)
                score = self.models[model].fit_score_partial(X_model)
                score = self.postprocessor[model].fit_transform_partial(score)
                data[model + '_score'] = np.nan_to_num(score) * 100
            except Exception as e:
                self.info(X_model)
                self.info(e)
                self.info(f'prediction failed for {model} at run_counter {self.runs_counter}, using last prediction instead.')
                data[model + '_score'] = self.data_latest[model + '_score']

        return data

    def get_data(self):

        data = self.predict()
        self.data_latest = data

        self.validate_charts('score', data, divisor=100)

        return data