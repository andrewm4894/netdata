import io
import json
from contextlib import redirect_stdout

import pytest

from metric_correlations import run_metric_correlations
from netdata_pandas.data import get_chart_list

from insights_modules.model import chart_level_models, models_supported

# define some test inputs
test_host = 'london.my-netdata.io'
test_host_charts_available = set(get_chart_list(host=test_host))
min_result_len = 10


def do_test(host, model, model_level='dim'):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(host=host, model=model, model_level=model_level, run_mode='test')
    results = f.getvalue()
    results = json.loads(results)
    return results


def validate_results(results, model, model_level):
    results_data = results['data']
    charts_scored = set(results_data.keys())
    assert len(results_data) >= min_result_len
    assert charts_scored.issubset(test_host_charts_available)
    if model_level == 'chart' and model in chart_level_models:
        dims_list = [list(results_data[chart].keys()) for chart in results]
        dims_list_expected = [['*'] for i in range(len(charts_scored))]
        assert dims_list == dims_list_expected


@pytest.mark.parametrize("model_level", ['dim', 'chart'])
#@pytest.mark.parametrize("model", models_supported)
@pytest.mark.parametrize("model", ['ks'])
@pytest.mark.timeout(120)
def test_metric_correlations(model, model_level):
    results = do_test(host=test_host, model=model, model_level=model_level)
    validate_results(results, model, model_level)






