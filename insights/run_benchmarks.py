import argparse
import io
import re
import time
from contextlib import redirect_stdout

import pandas as pd
from metric_correlations import run_metric_correlations


def run_benchmarks(host=None, model_list=None, n_list=None):

    # parse args, arg may come in via command line or via a function call.
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, nargs='?', help='host', default='127.0.0.1:19999')
    parser.add_argument('--model_list', type=str, nargs='?', help='model_list', default='ks,hbos')
    parser.add_argument('--n_list', type=str, nargs='?', help='n_list', default='100,1000,5000,10000')
    args = parser.parse_args()
    host = args.host if not host else host
    model_list = args.model_list if not model_list else model_list
    n_list = args.n_list if not n_list else n_list

    model_list = model_list.split(',')
    n_list = [int(n) for n in n_list.split(',')]
    results_all = []
    now = time.time()

    for model in model_list:
        for n in n_list:
            print(f'... begin {model}, (n={n})')
            baseline_after = int(now - n)
            baseline_before = int(now - n/2)
            highlight_after = int(now - n/2)
            highlight_before = int(now)
            f = io.StringIO()
            with redirect_stdout(f):
                run_metric_correlations(
                    host=host, model=model, print_results=False, baseline_after=baseline_after,
                    baseline_before=baseline_before, highlight_after=highlight_after, highlight_before=highlight_before
                )
            results = f.getvalue()
            time_data = float(re.search(" (.*) seconds to get data", results).group(1))
            time_scores = float(re.search(" (.*) seconds to get scores", results).group(1))
            time_total = float(re.search(" (.*) seconds in total", results).group(1))
            results_all.append([model, n, time_data, time_scores, time_total])

    df_results = pd.DataFrame(results_all, columns=['model', 'n', 'time_data', 'time_scores', 'time_total'])
    print('---results---')
    print(df_results)


if __name__ == '__main__':
    run_benchmarks()

