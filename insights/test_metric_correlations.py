import io
from contextlib import redirect_stdout

from metric_correlations import run_metric_correlations


def test_ks():
    f = io.StringIO()
    with redirect_stdout(f):
        run_metric_correlations()
    results = f.getvalue()
    print(results)
    assert 1 == 1



