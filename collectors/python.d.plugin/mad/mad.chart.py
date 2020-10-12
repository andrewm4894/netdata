# -*- coding: utf-8 -*-
# Description: mad netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from datetime import datetime

import numpy as np
import pandas as pd

from bases.FrameworkServices.SimpleService import SimpleService
from netdata_pandas.data import get_data, get_allmetrics


priority = 50
update_every = 1

ORDER = [
    'mad',
]

CHARTS = {
    'mad': {
        'options': ['mad', 'mad', 'mad', 'mad', 'mad.mad', 'line'],
        'lines': []
    }
}


class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.host = self.configuration.get('host')
        self.charts_in_scope = self.configuration.get('charts_in_scope').split(',')
        self.train_secs = self.configuration.get('train_secs', 3600)
        self.offset_secs = self.configuration.get('offset_secs', 0)
        self.order = ORDER
        self.definitions = CHARTS

    @staticmethod
    def check():
        return True

    def validate_charts(self, name, data, algorithm='absolute', multiplier=1, divisor=1):
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def get_data(self):

        df = get_data(self.host, self.charts_in_scope, after=-self.train_secs, before=0, col_sep='.')
        data = df.mad().to_dict()

        self.validate_charts('mad', data)

        return data