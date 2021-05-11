# -*- coding: utf-8 -*-
# Description: autoencoder netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from json import loads
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from bases.FrameworkServices.UrlService import UrlService

update_every = 5
disabled_by_default = True

ORDER = [
    'scores',
]

CHARTS = {
    'scores': {
        'options': [None, 'autoencoder', 'score', 'Scores', 'scores', 'line'],
        'lines': []
    },
}

DEFAULT_PROTOCOL = 'http'
DEFAULT_HOST = '127.0.0.1:19999'


class AnomalyDetector(Model):
  def __init__(self, n_features):
    super(AnomalyDetector, self).__init__()
    self.n_features = n_features
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation='tanh'),
      layers.Dense(16, activation='tanh'),
      layers.Dense(8, activation='tanh'),
      ])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation='tanh'),
      layers.Dense(32, activation='tanh'),
      layers.Dense(self.n_features, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


def make_x(arr, lags_n=3, diffs_n=1, smooth_n=3):
    
        def lag(arr, n):
            res = np.empty_like(arr)
            res[:n] = np.nan
            res[n:] = arr[:-n]
            return res
        
        if diffs_n > 0:
            arr = np.diff(arr, diffs_n, axis=0)
            arr = arr[~np.isnan(arr).any(axis=1)]

        if smooth_n > 1:
            arr = np.cumsum(arr, axis=0, dtype=float)
            arr[smooth_n:] = arr[smooth_n:] - arr[:-smooth_n]
            arr = arr[smooth_n - 1:] / smooth_n

        if lags_n > 0:
            arr_orig = np.copy(arr)
            for lag_n in range(1,lags_n+1):
                arr = np.concatenate((arr, lag(arr_orig, lag_n)), axis=1)
            arr = arr[~np.isnan(arr).any(axis=1)]

        return arr


class Service(UrlService):
    def __init__(self, configuration=None, name=None):
        UrlService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.protocol = self.configuration.get('protocol', DEFAULT_PROTOCOL)
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.url = '{}://{}/api/v1/allmetrics?format=json'.format(self.protocol, self.host)
        self.charts_in_scope = ['system.cpu']
        self.collected_dims = {'scores': set()}
        self.train_data = {c:[] for c in self.charts_in_scope}
        self.pred_data = {c:[] for c in self.charts_in_scope}
        self.train_every = 10
        self.train_n = 10
        self.train_n_offset = 0
        self.model_last_fit = {c:0 for c in self.charts_in_scope}
        self.models = {c:None for c in self.charts_in_scope}
        self.n_features = {c:0 for c in self.charts_in_scope}
        self.lags_n = 3
        self.diffs_n = 1
        self.smooth_n = 3
        self.buffer_n = (self.lags_n + self.diffs_n + self.smooth_n) * 2

    def validate_charts(self, chart, data, algorithm='absolute', multiplier=1, divisor=1):
        """If dimension not in chart then add it.
        """
        if not self.charts:
            return

        for dim in data:
            if dim not in self.collected_dims[chart]:
                self.collected_dims[chart].add(dim)
                self.charts[chart].add_dimension([dim, dim, algorithm, multiplier, divisor])

        for dim in list(self.collected_dims[chart]):
            if dim not in data:
                self.collected_dims[chart].remove(dim)
                self.charts[chart].del_dimension(dim, hide=False)

    def _get_data(self):

        self.debug(self.runs_counter)

        # pull data from self.url
        raw_data = self._get_raw_data()
        if raw_data is None:
            return None

        raw_data = loads(raw_data)

        data_scores = {c: 0 for c in self.charts_in_scope}

        # process each chart
        for chart in self.charts_in_scope:

            x = [raw_data[chart]['dimensions'][dim]['value'] for dim in raw_data[chart]['dimensions']]
            
            if self.n_features[chart] == 0:
                
                self.n_features[chart] = len(x) + ( len(x) * self.lags_n )

            self.debug(x)

            self.train_data[chart].append(np.array(x))
            self.train_data[chart] = self.train_data[chart][-(self.train_n+self.train_n_offset):]
            self.train_data[chart] = self.train_data[chart][:self.train_n]

            self.pred_data[chart].append(np.array(x))
            #self.pred_data[chart] = self.pred_data[chart][-1]

            n_features = len(x)

            if self.models[chart] == None:

                self.models[chart] = AnomalyDetector(n_features=self.n_features[chart])
                self.models[chart].compile(optimizer='adam', loss='mae')

            if len(self.pred_data[chart]) > 0 and self.model_last_fit[chart] > 0:

                pred_data = make_x(np.array(self.train_data[chart][:self.buffer_n]), self.lags_n, self.diffs_n, self.smooth_n)[-1]
                self.debug(f'pred_data={pred_data}')
                pred_data = tf.cast(pred_data.reshape(1,-1), tf.float32)
                self.debug(f'pred_data.shape={pred_data.shape}')
                data_scores[chart] = np.mean(self.models[chart].predict(pred_data,steps=1))

            if self.runs_counter % self.train_every == 0 and len(self.train_data[chart]) >= self.train_n:

                train_data = make_x(np.array(self.train_data[chart]), self.lags_n, self.diffs_n, self.smooth_n)
                self.debug(f'train_data={train_data}')
                self.debug(f'train_data.shape={train_data.shape}')
                train_data = tf.cast(train_data, tf.float32)
                self.debug(f'train_data.shape={train_data.shape}')

                # fit model 
                history = self.models[chart].fit(train_data, train_data, 
                    epochs=5, 
                    batch_size=20,
                    shuffle=True,
                    verbose=0,
                    steps_per_epoch=10
                    )
                self.model_last_fit[chart] = self.runs_counter
                self.debug(f"{chart} model fit on {train_data.shape} data at {self.model_last_fit[chart]}, loss = {np.mean(history.history['loss'])}")

        self.validate_charts('scores', data_scores)

        data = {**data_scores}
        self.debug(data)

        return data
