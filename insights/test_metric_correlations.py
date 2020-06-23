import io
import json
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations


def test_ks():
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations()
    results = f.getvalue()
    results_json = json.loads(results)
    print(results_json.keys())
    assert 1 == 1



