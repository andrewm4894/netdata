# -*- coding: utf-8 -*-
# Description: aggregator netdata python.d module
# Author: andrewm4894
# SPDX-License-Identifier: GPL-3.0-or-later

import requests
import numpy as np

from bases.FrameworkServices.SimpleService import SimpleService

priority = 90000

ORDER = [
]

CHARTS = {
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
        self.charts_to_agg = {self.charts_to_agg[n]['name']: {'agg_func': self.charts_to_agg[n]['agg_func']} for n in range(0,len(self.charts_to_agg))}
        self.children = []
        self.parent_charts = self.get_charts()

    @staticmethod
    def check():
        return True

    def validate_charts(self, chart_name, data, algorithm='absolute', multiplier=1, divisor=1):
        chart_config = {'options': [None, 'A random number', 'random number', 'random', 'random', 'line']}
        if chart_name not in self.charts:
            chart_params = [chart_name] + chart_config['options']
            self.charts.add_chart(params=chart_params)
        for dim in data:
            if dim not in self.charts[chart_name]:
                self.charts[chart_name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def get_charts(self):
        r = requests.get(f'http://{self.parent}/api/v1/charts')
        return r.json().get('charts', {})

    def get_children(self):
        r = requests.get(f'http://{self.parent}/api/v1/info')
        return r.json().get('mirrored_hosts', {})

    def get_allmetrics(self, child):
        r = requests.get(f'http://{self.parent}/host/{child}/api/v1/allmetrics?format=json')
        return r.json()

    def get_data(self):

        # get children
        self.children = self.get_children()
        self.children = [child for child in self.children if self.child_contains in child]

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

            data = {}

            for chart in allmetrics_list:
                data_chart = {}
                out_chart = f"{self.out_prefix}_{chart.replace('.','_')}"
                for dim in allmetrics_list[chart]:
                    out_dim = f"{chart.replace('.','_')}_{dim}"
                    if self.charts_to_agg[chart]['agg_func'] == 'mean':
                        data_chart[out_dim] = np.mean(allmetrics_list[chart][dim])
                    else:
                        data_chart[out_dim] = np.mean(allmetrics_list[chart][dim])

                self.validate_charts(out_chart, data_chart)

                data = {**data, **data_chart}
                
        self.info(data)

        return data
