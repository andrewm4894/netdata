# -*- coding: utf-8 -*-
# Description: autoencoder netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from json import loads
import re

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
DEFAULT_CHARTS_REGEX = 'system.cpu'


class Service(UrlService):
    def __init__(self, configuration=None, name=None):
        UrlService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.protocol = self.configuration.get('protocol', DEFAULT_PROTOCOL)
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.url = '{}://{}/api/v1/allmetrics?format=json'.format(self.protocol, self.host)
        self.charts_regex = re.compile(self.configuration.get('charts_regex', DEFAULT_CHARTS_REGEX))
        self.charts_to_exclude = self.configuration.get('charts_to_exclude', '').split(',')
        self.collected_dims = {'scores': set()}

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

        # pull data from self.url
        raw_data = self._get_raw_data()
        if raw_data is None:
            return None

        raw_data = loads(raw_data)

        # filter to just the data for the charts specified
        charts_in_scope = list(filter(self.charts_regex.match, raw_data.keys()))
        charts_in_scope = [c for c in charts_in_scope if c not in self.charts_to_exclude]

        data_scores = {}

        # process each chart
        for chart in charts_in_scope:

            x = [raw_data[chart]['dimensions'][dim]['value'] for dim in raw_data[chart]['dimensions']]
            x = [x for x in x if x is not None]
            self.debug(x)

            data_scores[chart] = sum(x)

        self.validate_charts('scores', data_scores)

        data = {**data_scores}
        #self.debug(data)

        return data
