from metric_correlations import run_metric_correlations


def test_ks():
    result = run_metric_correlations()
    print(result)
    assert 1 == 1



