# -*- coding: utf-8 -*-
# Description: zscores netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from datetime import datetime
from random import SystemRandom

import requests
import numpy as np
import pandas as pd

from bases.FrameworkServices.SimpleService import SimpleService
from netdata_pandas.data import get_data, get_allmetrics


priority = 50
update_every = 1

DEFAULT_HOST = '127.0.0.1:19999'
DEFAULT_CHARTS_IN_SCOPE = 'system.cpu,system.load,system.io,system.pgpgio,system.ram,system.net,system.ip,system.ipv6,system.processes,system.ctxt,system.idlejitter,system.intr,system.softirqs,system.softnet_stat'
DEFAULT_TRAIN_SECS = 60*60*4
DEFAULT_OFFSET_SECS = 60*5
DEFAULT_TRAIN_EVERY_N = 60
DEFAULT_Z_SMOOTH_N = 10
DEFAULT_Z_CLIP = 10
DEFAULT_BURN_IN = 20

ORDER = [
    'zscores',
    'zscores_3sigma'
]

CHARTS = {
    'zscores': {
        'options': [None, 'Z Scores', 'name.chart', 'zscores', 'zscores.zscores', 'line'],
        'lines': []
    },
    'zscores_3sigma': {
        'options': [None, 'Z Scores >3 Sigma', 'name.chart', 'zscores', 'zscores.zscores_3sigma', 'stacked'],
        'lines': []
    },
}


class Service(SimpleService):
    
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.charts_in_scope = self.configuration.get('charts_in_scope', DEFAULT_CHARTS_IN_SCOPE).split(',')
        self.train_secs = self.configuration.get('train_secs', DEFAULT_TRAIN_SECS)
        self.offset_secs = self.configuration.get('offset_secs', DEFAULT_OFFSET_SECS)
        self.train_every_n = self.configuration.get('train_every_n', DEFAULT_TRAIN_EVERY_N)
        self.z_smooth_n = self.configuration.get('z_smooth_n', DEFAULT_Z_SMOOTH_N) 
        self.z_clip = self.configuration.get('z_clip', DEFAULT_Z_CLIP)
        self.burn_in = self.configuration.get('burn_in', DEFAULT_BURN_IN) 
        self.order = ORDER
        self.definitions = CHARTS
        self.random = SystemRandom()
        self.df_mean = pd.DataFrame()
        self.df_std = pd.DataFrame()
        self.df_z_history = pd.DataFrame()

    @staticmethod
    def check():
        return True

    def get_data(self):

        now = int(datetime.now().timestamp())
        after = now - self.offset_secs - self.train_secs
        before = now - self.offset_secs

        if self.runs_counter <= self.burn_in or self.runs_counter % self.train_every_n == 0:
            
            self.df_mean = get_data(self.host, charts=self.charts_in_scope, after=after, before=before, points=1, group='average', col_sep='.')
            self.df_mean = self.df_mean.transpose()
            self.df_mean.columns = ['mean']

            self.df_std = get_data(self.host, charts=self.charts_in_scope, after=after, before=before, points=1, group='stddev', col_sep='.')
            self.df_std = self.df_std.transpose()
            self.df_std.columns = ['std']
            self.df_std = self.df_std[self.df_std['std']>0]

        df_allmetrics = get_allmetrics(self.host, charts=self.charts_in_scope, wide=True, col_sep='.').transpose()

        df_z = pd.concat([self.df_mean, self.df_std, df_allmetrics], axis=1, join='inner')
        df_z['z'] = np.where(df_z['std'] > 0, (df_z['value'] - df_z['mean']) / df_z['std'], 0)
        df_z['z'] = df_z['z'].fillna(0).clip(lower=-self.z_clip, upper=self.z_clip)
        df_z_wide = df_z[['z']].reset_index().pivot_table(values='z', columns='index')

        self.df_z_history = self.df_z_history.append(df_z_wide, sort=True).tail(self.z_smooth_n)

        df_z_history_long = df_z_wide.melt(value_name='z')
        df_z_smooth = df_z_history_long.groupby('index')[['z']].mean() * 100
        df_z_smooth['3sig'] = np.where(abs(df_z_smooth['z']) > 300, 1, 0)
        
        df_z_smooth.index = ['.'.join(reversed(x.split('.'))) + '_z' for x in df_z_smooth.index]
        data_dict_z = df_z_smooth['z'].to_dict()
        
        df_z_smooth.index = [x[:-2] + '_3sig' for x in df_z_smooth.index]
        data_dict_3sig = df_z_smooth['3sig'].to_dict()

        data = {**data_dict_z, **data_dict_3sig}

        for dim in data_dict_z:
            if dim not in self.charts['zscores']:
                self.charts['zscores'].add_dimension([dim, dim, 'absolute', 1, 100])
        
        for dim in data_dict_3sig:
            if dim not in self.charts['zscores_3sigma']:
                self.charts['zscores_3sigma'].add_dimension([dim, dim, 'absolute', 1, 1])

        return data
