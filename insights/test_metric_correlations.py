import io
import json
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations
from netdata_pandas.data import get_chart_list

# define some test inputs
test_host = 'london.my-netdata.io'
test_host_charts_available = set(get_chart_list(host=test_host))


def do_test(model):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(model=model)
    results = f.getvalue()
    results = json.loads(results)
    return results


def test_ks():
    results = do_test('ks')
    charts_scored = set(results.keys())
    assert charts_scored.issubset(test_host_charts_available)



