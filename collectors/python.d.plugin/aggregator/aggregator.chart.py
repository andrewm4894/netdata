# -*- coding: utf-8 -*-
# Description: aggregator netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

import requests
import numpy as np

from bases.FrameworkServices.SimpleService import SimpleService

priority = 90000

ORDER = [
    'random',
]

CHARTS = {
    'random': {
        'options': [None, 'A random number', 'random number', 'random', 'random', 'line'],
        'lines': [
            ['random1']
        ]
    }
}


class Service(SimpleService):
    def __init__(self, configuration=None, name=None):
        SimpleService.__init__(self, configuration=configuration, name=name)
        self.order = ORDER
        self.definitions = CHARTS
        self.parent = self.configuration.get('parent', '127.0.0.1:19999')
        self.child_contains = self.configuration.get('child_contains', None)
        self.out_prefix = self.configuration.get('out_prefix', 'agg')
        self.charts_to_agg = self.configuration.get('charts_to_agg', None)
        self.children = []

    @staticmethod
    def check():
        return True

    def get_children(self):
        r = requests.get(f'http://{self.parent}/api/v1/info')
        return r.json().get('mirrored_hosts', {})

    def get_allmetrics(self, child):
        r = requests.get(f'http://{self.parent}/host/{child}/api/v1/allmetrics?format=json')
        return r.json()

    def get_data(self):

        self.info(self.charts_to_agg)
        XXX

        # get children
        self.children = self.get_children()
        self.children = [child for child in children if self.child_contains in child]

        if len(self.children) > 0:

            allmetrics = {}

            # get metrics from children
            for child in self.children:
                allmetrics_child = self.get_allmetrics(child)
                allmetrics[child] = {
                    allmetrics_child.get(chart, {}).get('name', ''): allmetrics_child.get(chart, {}).get('dimensions', {})
                    for chart in allmetrics_child if chart in self.charts_to_agg
                }

            # append metrics into a list
            allmetrics_list = {
                chart: {}
                for chart in set.union(*[set(allmetrics[child].keys()) for child in allmetrics])
            }
            for child in allmetrics:
                for chart in allmetrics[child]:
                    for dim in allmetrics[child][chart]:
                        if dim not in allmetrics_list[chart]:
                            allmetrics_list[chart][dim] = [allmetrics[child][chart][dim]['value']]
                        else:
                            allmetrics_list[chart][dim].append(allmetrics[child][chart][dim]['value'])

            # aggregate each metric over available data
            allmetrics_agg = {
                f"{out_prefix}.{chart.replace('.','_')}": {
                    dim: None
                    for dim in allmetrics_list[chart]
                }
                for chart in allmetrics_list
            }
            for chart in allmetrics_list:
                out_chart = f"{out_prefix}.{chart.replace('.','_')}"
                for dim in allmetrics_list[chart]:
                    if self.charts_to_agg[chart]['agg_func'] == 'mean':
                        allmetrics_agg[out_chart][dim] = np.mean(allmetrics_list[chart][dim])
                    else:
                        allmetrics_agg[out_chart][dim] = np.mean(allmetrics_list[chart][dim])

            #self.info(allmetrics_agg)

        data = dict()

        for i in range(1, 2):
            dimension_id = ''.join(['random', str(i)])

            if dimension_id not in self.charts['random']:
                self.charts['random'].add_dimension([dimension_id])

            data[dimension_id] = self.random.randint(0, 100)

        return data
