# -*- coding: utf-8 -*-
# Description: anomaliesonline netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

import time
from datetime import datetime
import re

import requests
import numpy as np
import pandas as pd
from netdata_pandas.data import get_allmetrics
from pysad.models import RobustRandomCutForest, xStream, KNNCAD, KitNet, LODA
from pysad.models import RobustRandomCutForest as DefaultModel
from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
from pysad.transform.postprocessing import RunningAveragePostprocessor

from bases.FrameworkServices.SimpleService import SimpleService


priority = 50
update_every: 2

ORDER = ['probability', 'anomaly']

CHARTS = {
    'probability': {
        'options': ['probability', 'Anomaly Probability', 'probability', 'anomaliesonline', 'anomaliesonline.probability', 'line'],
        'lines': []
    },
    'anomaly': {
        'options': ['anomaly', 'Anomaly', 'count', 'anomaliesonline', 'anomaliesonline.anomaly', 'stacked'],
        'lines': []
    },
}


class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.protocol = self.configuration.get('protocol', 'http')
        self.host = self.configuration.get('host', '127.0.0.1:19999')
        self.charts_regex = re.compile(self.configuration.get('charts_regex','system\..*'))
        self.charts_in_scope = list(filter(self.charts_regex.match, [c for c in requests.get(f'{self.protocol}://{self.host}/api/v1/charts').json()['charts'].keys()]))
        self.custom_models = self.configuration.get('custom_models', None)
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
        self.model = self.configuration.get('model', 'rrcf')
        self.lags_n = self.configuration.get('lags_n', 3)
        self.smooth_n = self.configuration.get('smooth_n', 3)
        self.diffs_n = self.configuration.get('diffs_n', 1)
        self.calibrator_window_size = self.configuration.get('calibrator_window_size', 1000)
        self.postprocessor_window_size = self.configuration.get('postprocessor_window_size', 15)
        if self.model == 'rrcf':
            self.models = {model: RobustRandomCutForest(num_trees=10, shingle_size=4, tree_size=256) for model in self.models_in_scope}
        elif self.model == 'xstream':
            self.models = {model: xStream(num_components=20, n_chains=20, depth=10, window_size=25) for model in self.models_in_scope}
        elif self.model == 'knncad':
            self.models = {model: KNNCAD(probationary_period=50) for model in self.models_in_scope}
        elif self.model == 'kitnet':
            self.models = {model: KitNet(max_size_ae=10, grace_feature_mapping=100, grace_anomaly_detector=100, learning_rate=0.1, hidden_ratio=0.75) for model in self.models_in_scope}
        elif self.model == 'loda':
            self.models = {model: LODA(num_bins=10, num_random_cuts=100) for model in self.models_in_scope}
        else:
            self.models = {model: DefaultModel() for model in self.models_in_scope}
        self.calibrators = {model: GaussianTailProbabilityCalibrator(running_statistics=True, window_size=self.calibrator_window_size) for model in self.models_in_scope}
        self.postprocessors = {model: RunningAveragePostprocessor(window_size=self.postprocessor_window_size) for model in self.models_in_scope}
        self.df = pd.DataFrame()
        self.data_latest = {}
        self.min_history = ((self.lags_n + 1) + (self.smooth_n + 1) + self.diffs_n)
        self.anomaly_threshold = float(self.configuration.get('anomaly_threshold', 90))
        self.raw_score_latest = {}

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        """If dimension not in chart then add it.
        """
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def make_features(self, arr, colnames):
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

        arr = np.nan_to_num(arr)

        return arr, colnames
    
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

    def predict(self):
        """Get latest data, make it into a feature vector, and get predictions for each available model.

        :return: (<dict>,<dict>) tuple of dictionaries, one for probability scores and the other for anomaly predictions.
        """
        data_probability, data_anomaly = {}, {}

        df_allmetrics = get_allmetrics(self.host, self.charts_in_scope, wide=True, sort_cols=True, protocol=self.protocol)
        if self.custom_models:
            df_allmetrics = self.add_custom_models_dims(df_allmetrics)

        # get latest data to predict on
        self.df = self.df.append(df_allmetrics, ignore_index=True, sort=True).tail(self.min_history).ffill()

        # make features
        X, feature_colnames = self.make_features(self.df.values, list(self.df.columns))

        # if no features then return empty data
        if len(X) == 0:
            return data_probability, data_anomaly

        # get scores on latest data
        X = X[-1].reshape(1,-1)
        data_probability, data_anomaly = self.try_predict(X, feature_colnames)

        return data_probability, data_anomaly

    def try_predict(self, X, feature_colnames):
        """Try make prediction and fall back to last known prediction if fails.

        :param X <np.ndarray>: feature vector.
        :param feature_colnames <list>: list of corresponding feature names.
        :return: (<dict>,<dict>) tuple of dictionaries, one for probability scores and the other for anomaly predictions.
        """
        data_probability, data_anomaly = {}, {}
        for model in self.models.keys():
            X_model = self.get_array_cols(feature_colnames, X, starts_with=model)
            try:
                score = self.models[model].fit_score_partial(X_model)
                if np.isnan(score):
                    score = self.raw_score_latest[model]
                self.raw_score_latest[model] = score
                if self.calibrator_window_size > 0:
                    score = self.calibrators[model].fit_transform(np.array([score]))
                if self.postprocessor_window_size > 0:
                    score = self.postprocessors[model].fit_transform_partial(score)
                score = np.mean(score) * 100
                data_probability[f'{model}_prob'] = score
            except Exception as e:
                self.info(X_model)
                self.info(e)
                self.info(f'prediction failed for {model} at run_counter {self.runs_counter}, using last prediction instead.')
                data_probability[model + '_prob'] = self.data_latest[model + '_prob']
                data_anomaly[model + '_anomaly'] = self.data_latest[model + '_anomaly']

        data_anomaly = {f"{k.replace('_prob','_anomaly')}": 1 if data_probability[k] >= self.anomaly_threshold else 0 for k in data_probability}

        return data_probability, data_anomaly

    def get_data(self):

        data_probability, data_anomaly = self.predict()
        data = {**data_probability, **data_anomaly}
        self.data_latest = data

        self.validate_charts('probability', data_probability, divisor=100)
        self.validate_charts('anomaly', data_anomaly)

        return data