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
from pysad.models import xStream, ExactStorm, HalfSpaceTrees, IForestASD, KitNet, KNNCAD, MedianAbsoluteDeviation, RobustRandomCutForest
from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler

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
        self.models_in_scope = self.charts_in_scope
        self.model = self.configuration.get('model', 'rrcf')
        self.lags_n = self.configuration.get('lags_n', 0)
        self.smooth_n = self.configuration.get('smooth_n', 0)
        self.diffs_n = self.configuration.get('diffs_n', 1)
        if self.model == 'xstream':
            self.models = {model: xStream() for model in self.models_in_scope}
        elif self.model == 'exact_storm':
            self.models = {model: ExactStorm() for model in self.models_in_scope}
        elif self.model == 'half_space_trees':
            self.models = {model: HalfSpaceTrees() for model in self.models_in_scope}
        elif self.model == 'iforest':
            self.models = {model: IForestASD() for model in self.models_in_scope}
        elif self.model == 'mad':
            self.models = {model: MedianAbsoluteDeviation() for model in self.models_in_scope}
        elif self.model == 'kitnet':
            self.models = {model: KitNet() for model in self.models_in_scope}
        elif self.model == 'rrcf':
            self.models = {model: RobustRandomCutForest() for model in self.models_in_scope}
        elif self.model == 'knncad':
            self.models = {model: KNNCAD(probationary_period=50) for model in self.models_in_scope}
        else:
            self.models = {model: xStream() for model in self.models_in_scope}
        self.calibrator_window_size = self.configuration.get('calibrator_window_size', 100)
        self.postprocessor_window_size = self.configuration.get('postprocessor_window_size', 10)
        self.calibrators = {model: GaussianTailProbabilityCalibrator(running_statistics=True, window_size=self.calibrator_window_size) for model in self.models_in_scope}
        self.postprocessors = {model: RunningAveragePostprocessor(window_size=self.postprocessor_window_size) for model in self.models_in_scope}
        self.data_latest = {}

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        """If dimension not in chart then add it.
        """
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def predict(self):
        """Get latest data, make it into a feature vector, and get predictions for each available model.

        :return: (<dict>,<dict>) tuple of dictionaries, one for probability scores and the other for anomaly predictions.
        """
        # get latest data to predict on
        df = get_allmetrics(self.host, self.charts_in_scope, wide=True, sort_cols=True)

        data = {}
        for model in self.models.keys():
            self.debug(f'{model}')
            X = df[df.columns[df.columns.str.startswith(model)]].values
            self.debug(X.shape)
            self.debug(X)
            score = self.models[model].fit_score_partial(X)
            if self.calibrator_window_size > 0:
                score = self.calibrators[model].fit_transform(np.array([score]))
            if self.postprocessor_window_size > 0:
                score = self.postprocessors[model].fit_transform_partial(score)
            score = np.mean(score) * 100
            data[f'{model}_score'] = score
        self.debug('data')
        self.debug(data)

        return data

    def get_data(self):

        data = self.predict()

        self.validate_charts('score', data, divisor=100)

        return data