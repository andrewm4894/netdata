import io
import json
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations


def do_test(model):
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations(model=model)
    results = f.getvalue()
    results = json.loads(results)
    return results


def test_ks():
    results = do_test('ks')
    print(results.keys())
    assert 1 == 1
    assert 1 == 2



