import io
import json
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations
from netdata_pandas.data import get_chart_list

# define some test inputs
test_host = 'london.my-netdata.io'
test_host_charts_available = set(get_chart_list(host=test_host))
min_result_len = 10


def do_test(host, model):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(host=host, model=model)
    results = f.getvalue()
    results = json.loads(results)
    return results


def validate_results(results):
    charts_scored = set(results.keys())
    assert len(results) >= min_result_len
    assert charts_scored.issubset(test_host_charts_available)


def test_ks_default():
    results = do_test(host=test_host, model='ks')
    validate_results(results)


def test_hbos_default():
    results = do_test(host=test_host, model='hbos')
    validate_results(results)


def test_knn_default():
    results = do_test(host=test_host, model='knn')
    validate_results(results)





