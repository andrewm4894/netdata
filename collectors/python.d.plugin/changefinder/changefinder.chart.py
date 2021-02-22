# -*- coding: utf-8 -*-
# Description: changefinder netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

from json import loads
import re

from bases.FrameworkServices.UrlService import UrlService

import numpy as np
import changefinder
from scipy.stats import percentileofscore

update_every = 1
disabled_by_default = True

ORDER = [
    'score',
    'flag'
]

CHARTS = {
    'score': {	
        'options': [None, 'ChangeFinder', 'score', 'ChangeFinder', 'score', 'line'],	
        'lines': [],	
        'variables': [	
            [],	
        ]	
    },
    'flag': {
        'options': [None, 'ChangeFinder', 'flag', 'ChangeFinder Flags', 'flag', 'stacked'],
        'lines': [],
        'variables': [
            [],
        ]
    }
}

DEFAULT_PROTOCOL = 'http'
DEFAULT_HOST = '127.0.0.1:19999'
DEFAULT_CHARTS_REGEX = 'system.*'
DEFAULT_MODE = 'per_chart'
DEFAULT_CF_R = 0.5
DEFAULT_CF_ORDER = 1
DEFAULT_CF_SMOOTH = 15
DEFAULT_CF_DIFF = False
DEFAULT_CF_THRESHOLD = 99
DEFAULT_N_SAMPLES = 3600
DEFAULT_SHOW_SCORES = False


class Service(UrlService):
    def __init__(self, configuration=None, name=None):
        UrlService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.host = self.configuration.get('host', DEFAULT_HOST)
        self.protocol = self.configuration.get('protocol', DEFAULT_PROTOCOL)
        self.charts_regex = re.compile(self.configuration.get('charts_regex', DEFAULT_CHARTS_REGEX))
        self.mode = self.configuration.get('mode', DEFAULT_MODE)
        self.n_samples = int(self.configuration.get('n_samples', DEFAULT_N_SAMPLES))
        self.show_scores = int(self.configuration.get('show_scores', DEFAULT_SHOW_SCORES))
        self.cf_r = float(self.configuration.get('cf_r', DEFAULT_CF_R))
        self.cf_order = int(self.configuration.get('cf_order', DEFAULT_CF_ORDER))
        self.cf_smooth = int(self.configuration.get('cf_smooth', DEFAULT_CF_SMOOTH))
        self.cf_diff = bool(self.configuration.get('cf_diff', DEFAULT_CF_DIFF))
        self.cf_threshold = float(self.configuration.get('cf_threshold', DEFAULT_CF_THRESHOLD))
        self.url = '{}://{}/api/v1/allmetrics?format=json'.format(self.protocol, self.host)
        self.models = {}
        self.x_latest = {}
        self.scores_latest = {}
        self.scores_samples = {}

    def get_score(self, x, model):
        """Update the score for the model based on most recent data, flag if it's percentile passes self.cf_threshold.
        """

        # get score
        if model not in self.models:
            # initialise empty model if needed
            self.models[model] = changefinder.ChangeFinder(r=self.cf_r, order=self.cf_order, smooth=self.cf_smooth)
        # if the update for this step fails then just fallback to last known score
        try:
            score = self.models[model].update(x)
            self.scores_latest[model] = score
        except:
            score = self.scores_latest.get(model, 0)        
        score = 0 if np.isnan(score) else score

        # update sample scores
        if model in self.scores_samples:
            self.scores_samples[model].append(score)
        else:
            self.scores_samples[model] = [score]
        self.scores_samples[model] = self.scores_samples[model][-self.n_samples:]

        # convert score to percentile
        score = percentileofscore(self.scores_samples[model], score)
        
        # flag based on score percentile
        flag = 1 if score >= self.cf_threshold else 0

        return score, flag

    def update_chart(self, chart, data, algo='absolute', multiplier=1, divisor=1):
        for dim in data:
            if dim not in self.charts[chart]:
                self.charts[chart].add_dimension([dim, dim, algo, multiplier, divisor])

    def _get_data(self):
        raw_data = self._get_raw_data()
        if raw_data is None:
            return None

        raw_data = loads(raw_data)
        charts = list(filter(self.charts_regex.match, raw_data.keys()))
        data_score = {}
        data_flag = {}

        for chart in charts:

            if self.mode == 'per_chart':

                x = [raw_data[chart]['dimensions'][x]['value'] for x in raw_data[chart]['dimensions']]
                x = [x for x in x if x is not None]
                x = sum(x) / len(x)
                if self.cf_diff:
                    x_diff = x - self.x_latest.get(chart, 0)
                    self.x_latest[chart] = x
                    x = x_diff
                score, flag = self.get_score(x, chart)
                data_score['{}_score'.format(chart)] = score * 100
                data_flag[chart] = flag

            else:

                for dim in raw_data[chart]['dimensions']:

                    x = raw_data[chart]['dimensions'][dim]['value']
                    x = x if x else 0
                    dim = '{}|{}'.format(chart, dim)
                    if self.cf_diff:
                        x_diff = x - self.x_latest.get(dim, 0)
                        self.x_latest[dim] = x
                        x = x_diff
                    score, flag = self.get_score(x, dim)
                    data_score['{}_score'.format(dim)] = score * 100
                    data_flag[dim] = flag

        self.update_chart('score', data_score, divisor=100)
        self.update_chart('flag', data_flag)

        data = {**data_score, **data_flag}

        return data
    
