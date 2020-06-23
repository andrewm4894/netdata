import io
import json
from contextlib import redirect_stdout

import pytest

from metric_correlations import run_metric_correlations
from netdata_pandas.data import get_chart_list

# define some test inputs
test_host = 'london.my-netdata.io'
test_host_charts_available = set(get_chart_list(host=test_host))
min_result_len = 10


def do_test(host, model, model_level='dim'):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(host=host, model=model, model_level=model_level)
    results = f.getvalue()
    results = json.loads(results)
    return results


def validate_results(results, model_level):
    charts_scored = set(results.keys())
    assert len(results) >= min_result_len
    assert charts_scored.issubset(test_host_charts_available)
    if model_level == 'chart':
        print('------')
        print([list(results[chart].keys()) for chart in results])
        print('------')
        print([['*'] for i in range(len(charts_scored))])
        print('------')
        assert [list(results[chart].keys()) for chart in results] == [['*'] for i in range(len(charts_scored))]


@pytest.mark.parametrize("model", ['ks', 'hbos'])
@pytest.mark.parametrize("model_level", ['dim', 'chart'])
def test_metric_correlations(model, model_level):
    results = do_test(host=test_host, model=model, model_level=model_level)
    validate_results(results, model_level)






