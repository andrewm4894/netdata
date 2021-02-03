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
        self.definitions = CHARTS
        self.parent = self.configuration.get('parent', '127.0.0.1:19999')
        self.child_contains = self.configuration.get('child_contains', None)
        self.out_prefix = self.configuration.get('out_prefix', 'agg')
        self.charts_to_agg = self.configuration.get('charts_to_agg', None)
        self.charts_to_agg = {
            self.charts_to_agg[n]['name']: {
                'agg_func': self.charts_to_agg[n].get('agg_func','mean'),
                'exclude_dims': self.charts_to_agg[n].get('exclude_dims','').split(',')
                } 
                for n in range(0,len(self.charts_to_agg))
        }
        self.refresh_children_every_n = self.configuration.get('refresh_children_every_n', 60)
        self.children = []
        self.parent_charts = self.get_charts()
        self.allmetrics = {}
        self.allmetrics_list = {c: {} for c in self.charts_to_agg}

    def check(self):
        if len(self.get_children()) >= 1:
            return True
        else:
            return False

    def validate_charts(self, name, data, title, units, family, context, chart_type='line', algorithm='absolute', multiplier=1, divisor=1):
        config = {'options': [name, title, units, family, context, chart_type]}
        if name not in self.charts:
            params = [name] + config['options']
            self.charts.add_chart(params=params)
        for dim in data:
            if dim not in self.charts[name]:
                self.charts[name].add_dimension([dim, dim, algorithm, multiplier, divisor])

    def get_charts(self):
        r = requests.get(f'http://{self.parent}/api/v1/charts')
        return r.json().get('charts', {})

    def get_children(self):
        r = requests.get(f'http://{self.parent}/api/v1/info')
        return r.json().get('mirrored_hosts', {})

    def get_allmetrics(self, child):
        r = requests.get(f'http://{self.parent}/host/{child}/api/v1/allmetrics?format=json')
        return r.json()

    def scrape_children(self):
        for child in self.children:
            allmetrics_child = self.get_allmetrics(child)
            self.allmetrics[child] = {
                allmetrics_child.get(chart, {}).get('name', ''): allmetrics_child.get(chart, {}).get('dimensions', {})
                for chart in allmetrics_child if chart in self.charts_to_agg
            }

    def append_metrics(self):
        for child in self.allmetrics:
            for chart in self.allmetrics[child]:
                for dim in self.allmetrics[child][chart]:
                    if dim not in self.charts_to_agg[chart]['exclude_dims']:
                        if dim not in self.allmetrics_list[chart]:
                            self.allmetrics_list[chart][dim] = [self.allmetrics[child][chart][dim]['value']]
                        else:
                            self.allmetrics_list[chart][dim].append(self.allmetrics[child][chart][dim]['value'])

    def get_data(self):

        # get children
        if self.children == [] or self.runs_counter % self.refresh_children_every_n == 0:
            self.children = self.get_children()
            self.children = [child for child in self.children if self.child_contains in child]

        # process data if we have children that match
        if len(self.children) > 0:

            self.scrape_children()

            self.append_metrics()            

            data = {}

            for chart in self.allmetrics_list:
                data_chart = {}
                out_chart = f"{chart.replace('.','_')}"
                for dim in self.allmetrics_list[chart]:
                    out_dim = f"{chart.replace('.','_')}_{dim}"
                    if self.charts_to_agg[chart]['agg_func'] == 'mean':
                        data_chart[out_dim] = np.mean(self.allmetrics_list[chart][dim])*1000
                    else:
                        data_chart[out_dim] = np.mean(self.allmetrics_list[chart][dim])*1000

                self.validate_charts(
                    name=out_chart, 
                    title=out_chart, 
                    units=self.parent_charts[chart].get('units',''), 
                    family=chart.replace('.','_'), 
                    context=out_chart, 
                    chart_type=self.parent_charts[chart].get('chart_type','line'), 
                    data=data_chart,
                    divisor=1000
                )

                data = {**data, **data_chart}
        
        self.allmetrics_list = {c: {} for c in self.charts_to_agg}

        return data
