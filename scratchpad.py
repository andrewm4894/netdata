#%%

from random import SystemRandom, randint

from bases.FrameworkServices.SimpleService import SimpleService

priority = 90000

ORDER = [
    'cpu',
    'load',
    'io',
    'net',
    'scores',
    'anomalies',
]

CHARTS = {
    'cpu': {
        'options': [None, 'CPU Anomaly Scores', 'value', 'cpu', 'anomalies.cpu', 'line'],
        'lines': [
            ['cpu_expected'], ['cpu_actual'], ['cpu_error']
        ]
    },
    'load': {
        'options': [None, 'Load Anomaly Scores', 'value', 'load', 'anomalies.load', 'line'],
        'lines': [
            ['load_expected'], ['load_actual'], ['load_error']
        ]
    },
    'io': {
        'options': [None, 'IO Anomaly Scores', 'value', 'io', 'anomalies.io', 'line'],
        'lines': [
            ['io_expected'], ['io_actual'], ['io_error']
        ]
    },
    'net': {
        'options': [None, 'Network Anomaly Scores', 'value', 'net', 'anomalies.net', 'line'],
        'lines': [
            ['net_expected'], ['net_actual'], ['net_error']
        ]
    },
    'scores': {
        'options': [None, 'All Anomaly Scores', 'value', 'scores', 'anomalies.scores', 'line'],
        'lines': [
            ['cpu_score'], ['load_score'], ['io_score'], ['net_score']
        ]
    },
    'anomalies': {
        'options': [None, 'All Anomaly Events', 'is_anomaly', 'anomalies', 'anomalies.anomalies', 'line'],
        'lines': [
            ['cpu_anomaly'], ['load_anomaly'], ['io_anomaly'], ['net_anomaly']
        ]
    },
}

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 19999

def get_data():

    data = dict()

    for chart in ['cpu', 'load', 'io', 'net']:

        after = -1
        url = f'https://{DEFAULT_HOST}:{DEFAULT_PORT}/api/v1/data?chart=system.{chart}&after={after}&format=json'
        response = requests.get(url)
        raw_data = response.json()['data'][0][1:]

        actual = np.mean(raw_data)
        rand_error_pct = randint(-10, 10) / 100
        expected = actual + (rand_error_pct * actual)
        error = abs(actual - expected)
        error_pct = error / actual
        anomalies = 1 if error_pct > 0.01 else 0

        data[f'{chart}_expected'] = expected
        data[f'{chart}_actual'] = actual
        data[f'{chart}_error'] = error
        data[f'{chart}_score'] = error
        data[f'{chart}_anomaly'] = anomalies

    return data

#%%

get_data()

#%%

import numpy as np
import pandas as pd
import requests




#%%

5 / 100

#%%

(randint(-10, 10) / 100)

#%%