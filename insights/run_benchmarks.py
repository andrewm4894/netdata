import argparse
import io
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations


def run_benchmarks():

    results = []

    for model in ['ks', 'hbos']:
        print("--------------------")
        print(model)
        f = io.StringIO()
        with redirect_stdout(f):
            run_metric_correlations(host='london.my-netdata.io', model=model, print_results=False)
        results.append(f.getvalue())

    print('---results---')
    print('---results[0]---')
    print(results[0])
    print('---results[1]---')
    print(results[1])

    print('---dev---')
    for line in results[1]:
        print(line)


#    # parse args
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--n', type=str, nargs='?', help='n', default='120')
#    args = parser.parse_args()
#    n = args.n
#    print(n)
#
#


if __name__ == '__main__':
    run_benchmarks()

