# -*- coding: utf-8 -*-
# Description: changefinder netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from json import loads
import re

from bases.FrameworkServices.UrlService import UrlService

import numpy as np
import changefinder

update_every = 2
disabled_by_default = True


def charts_template():
    order = [
        'changefinder',
    ]

    charts = {
        'changefinder': {
            'options': [None, 'ChangeFinder', 'score', 'changefinder', 'score', 'line'],
            'lines': [],
            'variables': [
                [],
            ]
        }
    }
    return order, charts


DEFAULT_PROTOCOL = 'http'
DEFAULT_HOST = '127.0.0.1:19999'
DEFAULT_CHARTS_REGEX = 'system.*'
DEFAULT_MODE = 'per_chart'


class Service(UrlService):
    def __init__(self, configuration=None, name=None):
        UrlService.__init__(self, configuration=configuration, name=name)
        self.order, self.definitions = charts_template()
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.protocol = self.configuration.get('protocol', DEFAULT_PROTOCOL)
        self.charts_regex = re.compile(self.configuration.get('charts_regex', DEFAULT_CHARTS_REGEX))
        self.mode = self.configuration.get('mode', DEFAULT_MODE)
        self.url = '{}://{}/api/v1/allmetrics?format=json'.format(self.protocol, self.host)
        self.models = {}
        self.min = {}
        self.max = {}
        self.models = {}

    def update_min(self, model, score):
        if model not in self.min:
            self.min[model] = score
        else:
            if score < self.min[model]:
                self.min[model] = score

    def update_max(self, model, score):
        if model not in self.max:
            self.max[model] = score
        else:
            if score > self.max[model]:
                self.max[model] = score

    def get_score(self, x, model):
        if model not in self.models:
            self.models[model] = changefinder.ChangeFinder()
        score = self.models[model].update(x)
        score = 0 if np.isnan(score) else score
        if self.max.get(model, 1) == 0:
            score = 0
        else:
            score = ( score - self.min.get(model, 0) ) / ( self.max.get(model, 1) - self.min.get(model, 0) )
        self.update_min(model, score)
        self.update_max(model, score)

        return score

    def update_chart(self, chart, data):
        if not self.charts:
            return

        for dim in data:
            if dim not in self.charts[chart]:
                self.charts[chart].add_dimension([dim, dim, 'absolute', '1', 100])

    def _get_data(self):
        raw_data = self._get_raw_data()
        if raw_data is None:
            return None

        raw_data = loads(raw_data)
        charts = list(filter(self.charts_regex.match, raw_data.keys()))
        data = {}

        for chart in charts:

            if self.mode == 'per_chart':

                x = [raw_data[chart]['dimensions'][x]['value'] for x in raw_data[chart]['dimensions']]
                x = [x for x in x if x is not None]
                x = sum(x) / len(x)
                score = self.get_score(x, chart)
                data[chart] = score * 100

            else:

                for dim in raw_data[chart]['dimensions']:

                    x = raw_data[chart]['dimensions'][dim]['value']
                    x = x if x else 0
                    dim = '{}|{}'.format(chart, dim)
                    score = self.get_score(x, dim)
                    data[dim] = score * 100
        
        self.update_chart('changefinder', data)

        return data
    
