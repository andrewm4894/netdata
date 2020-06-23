"""
Python script to calculate metric correlations.
"""
import argparse
import json
import logging
import sys
import time

from netdata_pandas.data import get_data, get_chart_list
from insights_modules.model import run_model
from insights_modules.utils import normalize_results


def run_metric_correlations(host=None, baseline_after=None, baseline_before=None, highlight_after=None,
                            highlight_before=None, model=None, n_lags=None, log_level=None, results_file=None,
                            max_points=None, print_results=None, model_errors='fail', model_level=None):

    time_start = time.time()

    # parse args, arg may come in via command line or via a function call.
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, nargs='?', help='host', default='127.0.0.1:19999')
    parser.add_argument('--baseline_after', type=str, nargs='?', help='baseline_after', default='-240')
    parser.add_argument('--baseline_before', type=str, nargs='?', help='baseline_before', default='-120')
    parser.add_argument('--highlight_after', type=str, nargs='?', help='highlight_after', default='-120')
    parser.add_argument('--highlight_before', type=str, nargs='?', help='highlight_before', default='0')
    parser.add_argument('--model', type=str, nargs='?', help='model', default='ks')
    parser.add_argument('--n_lags', type=str, nargs='?', help='n_lags', default='2')
    parser.add_argument('--log_level', type=str, nargs='?', help='log_level', default='info')
    parser.add_argument('--results_file', type=str, nargs='?', help='results_file', default=None)
    parser.add_argument('--max_points', type=str, nargs='?', help='max_points', default='10000')
    parser.add_argument('--print_results', type=bool, nargs='?', help='print_results', default=True)
    parser.add_argument('--model_errors', type=bool, nargs='?', help='model_errors', default='fail')
    parser.add_argument('--model_level', type=str, nargs='?', help='model_level', default='dim')
    args, unknown = parser.parse_known_args()
    host = args.host if not host else host
    baseline_after = int(args.baseline_after) if not baseline_after else int(baseline_after)
    baseline_before = int(args.baseline_before) if not baseline_before else int(baseline_before)
    highlight_after = int(args.highlight_after) if not highlight_after else int(highlight_after)
    highlight_before = int(args.highlight_before) if not highlight_before else int(highlight_before)
    model = args.model if not model else model
    n_lags = int(args.n_lags) if not n_lags else int(n_lags)
    log_level = args.log_level if not log_level else log_level
    results_file = args.results_file if not results_file else results_file
    max_points = int(args.max_points) if not max_points else int(max_points)
    print_results = args.print_results if print_results is None else print_results
    model_errors = args.model_errors if model_errors is None else model_errors
    model_level = args.model_level if model_level is None else model_level

    # set up logging
    if log_level == 'info':
        logging.basicConfig(level=logging.INFO)
    elif log_level == 'debug':
        logging.basicConfig(level=logging.DEBUG)
    elif log_level == 'warn':
        logging.basicConfig(level=logging.WARN)
    elif log_level == 'error':
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.WARN)
    log = logging.getLogger(__name__)
    #log.addHandler(logging.StreamHandler(sys.stdout))

    # handle 'after' and 'before' values if passed in as relative integers
    if baseline_after <= 0:
        baseline_after = int(time_start + baseline_after)
    if baseline_before <= 0:
        baseline_before = int(time_start + baseline_before)
    if highlight_after <= 0:
        highlight_after = int(time_start + highlight_after)
    if highlight_before <= 0:
        highlight_before = int(time_start + highlight_before)

    # limit the number of points we ask the rest api for
    points_expected = highlight_before - baseline_after
    if points_expected > max_points:
        points = max_points
    else:
        points = 0

    log.debug(f"... host={host}")
    log.debug(f"... baseline_after={baseline_after}")
    log.debug(f"... baseline_before={baseline_before}")
    log.debug(f"... highlight_after={highlight_after}")
    log.debug(f"... highlight_before={highlight_before}")
    log.debug(f"... model={model}")
    log.debug(f"... n_lags={n_lags}")
    log.debug(f"... log_level={log_level}")
    log.debug(f"... max_points={max_points}")
    log.debug(f"... results_file={results_file}")
    log.debug(f"... model_errors={model_errors}")
    log.debug(f"... model_level={model_level}")

    # get charts
    charts = get_chart_list(host)

    log.info(f"... num charts available on host={len(charts)}")

    # get data
    df = get_data(host, charts, after=baseline_after, before=highlight_before, diff=True, points=points,
                  ffill=True, numeric_only=True, nunique_thold=0.05, col_sep='|')

    log.info(f"... num charts in df={len(set([col.split('|')[0] for col in df.columns]))}")
    log.info(f"... df.shape={df.shape}")
    log.debug(f"... df.head()={df.head()}")

    # get numpy arrays
    colnames = list(df.columns)
    arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
    arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
    charts = list(set([col.split('|')[0] for col in colnames]))

    log.debug(f'... charts = {charts}')
    log.debug(f'... colnames = {colnames}')
    log.debug(f'... arr_baseline.shape = {arr_baseline.shape}')
    log.debug(f'... arr_highlight.shape = {arr_highlight.shape}')

    time_got_data = time.time()
    log.info(f'... {round(time_got_data - time_start,2)} seconds to get data.')

    # get scores
    results = run_model(model, colnames, arr_baseline, arr_highlight, n_lags, model_errors=model_errors, model_level=model_level)

    # normalize results
    results = normalize_results(results)

    time_got_scores = time.time()
    log.info(f'... {round(time_got_scores - time_got_data,2)} seconds to get scores.')

    if results_file:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    time_done = time.time()
    log.info(f'... {round(time_done - time_start, 2)} seconds in total.')

    if print_results:
        print(results)



if __name__ == '__main__':
    run_metric_correlations()


