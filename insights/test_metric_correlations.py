import io
import json
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations
from netdata_pandas.data import get_chart_list

# define some test inputs
test_host = 'london.my-netdata.io'
test_host_charts_available = get_chart_list(host=test_host)


def do_test(model):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(model=model)
    results = f.getvalue()
    results = json.loads(results)
    return results


def test_ks():
    results = do_test('ks')
    print(set(results.keys()))
    print(set(test_host_charts_available))
    assert 1 == 1
    assert 1 == 2



