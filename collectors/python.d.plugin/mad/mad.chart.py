# -*- coding: utf-8 -*-
# Description: mad netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from datetime import datetime

import requests
import numpy as np
import pandas as pd
from netdata_pandas.data import get_data, get_allmetrics
from pyod.models.mad import MAD

from bases.FrameworkServices.SimpleService import SimpleService

np.seterr(divide='ignore', invalid='ignore')

priority = 50

ORDER = ['probability', 'anomaly']

CHARTS = {
    'probability': {
        'options': ['probability', 'MAD Anomaly Probability', 'probability', 'mad', 'mad.probability', 'line'],
        'lines': []
    },
    'anomaly': {
        'options': ['anomaly', 'MAD Anomaly', 'count', 'mad', 'mad.anomaly', 'stacked'],
        'lines': []
    },
}


class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.host = self.configuration.get('host', '127.0.0.1:19999')
        if self.configuration.get('charts_in_scope','system.*') == 'system.*':
            self.charts_in_scope = [c for c in requests.get(f'http://{self.host}/api/v1/charts').json()['charts'].keys() if c.startswith('system.')]
        else:
            self.charts_in_scope = self.configuration.get('charts_in_scope').split(',')
        self.train_n_secs = self.configuration.get('train_n_secs', 3600)
        self.offset_n_secs = self.configuration.get('offset_n_secs', 0)
        self.train_every_n = self.configuration.get('train_every_n', 300)
        self.smooth_n = self.configuration.get('smooth_n', 0)
        self.diffs_n = self.configuration.get('diffs_n', 0)
        self.fitted = False
        self.df_predict = pd.DataFrame()
        self.dims_in_scope = []

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def make_features(self, df):
        if self.diffs_n >= 1:
            df = df.diff(self.diffs_n).dropna()
        if self.smooth_n >= 2:
            df = df.rolling(self.smooth_n).mean().dropna()
        return df

    def get_data(self):
        
        # define train data range
        before = int(datetime.now().timestamp()) - self.offset_n_secs
        after =  before - self.train_n_secs

        # train
        if not(self.fitted) or self.runs_counter % self.train_every_n == 0:
            df_train = self.make_features(get_data(self.host, self.charts_in_scope, after=after, before=before))
            self.dims_in_scope = list(df_train.columns)
            self.models = {dim: MAD() for dim in self.dims_in_scope}
            for dim in self.dims_in_scope:
                X = df_train[[dim]].dropna().values
                self.models[dim] = self.models[dim].fit(X)
            self.fitted = True

        # predict
        data_probability, data_anomaly = {}, {}
        self.df_predict = self.make_features(
            get_data(self.host, self.charts_in_scope, after=-(self.smooth_n + self.diffs_n) * 2, before=0)
            ).tail(1)
        for dim in self.dims_in_scope:
            X = self.df_predict[[dim]].dropna().values
            data_probability[dim + '_prob'] = np.nan_to_num(self.models[dim].predict_proba(X)[-1][1]) * 100
            data_anomaly[dim + '_anomaly'] = self.models[dim].predict(X)[-1]
        data = {**data_probability, **data_anomaly}

        self.validate_charts('probability', data_probability, divisor=100)
        self.validate_charts('anomaly', data_anomaly)

        return data