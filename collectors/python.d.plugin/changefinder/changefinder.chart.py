# -*- coding: utf-8 -*-
# Description: changefinder netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from json import loads
import re

from bases.FrameworkServices.UrlService import UrlService

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
DEFAULT_CHARTS_REGEX = 'system.cpu'
DEFAULT_MODE = 'per_dim'


class Service(UrlService):
    def __init__(self, configuration=None, name=None):
        UrlService.__init__(self, configuration=configuration, name=name)
        self.order, self.definitions = charts_template()
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.protocol = self.configuration.get('protocol', DEFAULT_PROTOCOL)
        self.charts_regex = re.compile(self.configuration.get('charts_regex', DEFAULT_CHARTS_REGEX))
        self.url = '{}://{}/api/v1/allmetrics?format=json'.format(self.protocol, self.host)
        self.models = {}
        self.mode = self.configuration.get('mode', DEFAULT_MODE)
        

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

                if chart not in self.models:
                    self.models[chart] = changefinder.ChangeFinder()

                score, _ = self.models[chart].update(x)
                data[chart] = score * 100

            else:

                for dim in raw_data[chart]['dimensions']:

                    model = '{}|{}'.format(chart,dim)

                    x = raw_data[chart]['dimensions'][dim]['value']

                    self.info(model)
                    self.info(x)

                    if model not in self.models:
                        self.models[model] = changefinder.ChangeFinder()                        

                    if x is not None:

                        score, _ = self.models[model].update(x)
                        data[model] = score * 100
        
        self.update_chart('changefinder', data)

        return data

    def update_chart(self, chart, data):
        if not self.charts:
            return

        for dim in data:
            if dim not in self.charts[chart]:
                self.charts[chart].add_dimension([dim, dim, 'absolute', '1', 100])
