# -*- coding: utf-8 -*-
# Description: anomalies_online netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

import time
from datetime import datetime
import re

import requests
import numpy as np
import pandas as pd
from netdata_pandas.data import get_data, get_allmetrics
from pysad.models import RobustRandomCutForest
from pysad.models import RobustRandomCutForest as DefaultModel
from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
from pysad.transform.postprocessing import RunningAveragePostprocessor

from bases.FrameworkServices.SimpleService import SimpleService


priority = 50
update_every: 2

ORDER = ['probability', 'anomaly']

CHARTS = {
    'probability': {
        'options': ['probability', 'Anomaly Probability', 'probability', 'anomalies_online', 'anomalies_online.probability', 'line'],
        'lines': []
    },
    'anomaly': {
        'options': ['anomaly', 'Anomaly', 'count', 'anomalies', 'anomalies_online.anomaly', 'stacked'],
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
        self.calibrator_window_size = self.configuration.get('calibrator_window_size', 100)
        self.postprocessor_window_size = self.configuration.get('postprocessor_window_size', 10)
        if self.model == 'rrcf':
            self.models = {model: RobustRandomCutForest() for model in self.models_in_scope}
        else:
            self.models = {model: DefaultModel() for model in self.models_in_scope}
        self.calibrators = {model: GaussianTailProbabilityCalibrator(running_statistics=True, window_size=self.calibrator_window_size) for model in self.models_in_scope}
        self.postprocessors = {model: RunningAveragePostprocessor(window_size=self.postprocessor_window_size) for model in self.models_in_scope}
        self.df = pd.DataFrame()
        self.data_latest = {}
        self.min_history = ((self.lags_n + 1) + (self.smooth_n + 1) + self.diffs_n)
        self.anomaly_threshold = float(self.configuration.get('anomaly_threshold', 90))

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        """If dimension not in chart then add it.
        """
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def make_features(self, df):
        """Process dataframe to add lags, smoothing or differences.

        :return: <pd.DataFrame> dataframe with preprocessing done.
        """
        if self.diffs_n >= 1:
            df = df.diff(self.diffs_n).dropna()
        if self.smooth_n >= 2:
            df = df.rolling(self.smooth_n).mean().dropna()
        if self.lags_n >= 1:
            df = pd.concat([df.shift(n) for n in range(self.lags_n + 1)], axis=1).dropna()
        return df

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

        df_allmetrics = get_allmetrics(self.host, self.charts_in_scope, wide=True, sort_cols=True)
        if self.custom_models:
            df_allmetrics = self.add_custom_models_dims(df_allmetrics)

        # get latest data to predict on
        self.df = self.df.append(df_allmetrics, ignore_index=True, sort=True).tail(self.min_history).ffill()

        # make features
        df = self.make_features(self.df).tail(1)

        # if no features then return empty data
        if len(df) == 0:
            return data_probability, data_anomaly

        # get scores
        for model in self.models.keys():
            X = df[df.columns[df.columns.str.startswith(model)]].values
            score = self.models[model].fit_score_partial(X)
            if self.calibrator_window_size > 0:
                score = self.calibrators[model].fit_transform(np.array([score]))
            if self.postprocessor_window_size > 0:
                score = self.postprocessors[model].fit_transform_partial(score)
            score = np.mean(score) * 100
            data_probability[f'{model}_prob'] = score
        
        # get anomaly flags
        data_anomaly = {f"{k.replace('_prob','_anomaly')}": 1 if data_probability[k] >= self.anomaly_threshold else 0 for k in data_probability}

        return data_probability, data_anomaly

    def get_data(self):

        data_probability, data_anomaly = self.predict()
        data = {**data_probability, **data_anomaly}

        self.validate_charts('probability', data_probability, divisor=100)
        self.validate_charts('anomaly', data_anomaly)

        return data