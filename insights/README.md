## Insights

Statistics and machine learning based "Insights" features.

### metric_correlations

Example usage:

```
python metric_correlations.py \
  --host='127.0.0.1:19999' \
  --baseline_after='-480' \
  --baseline_before='-240' \
  --highlight_after='-240' \
  --highlight_before='0' \
  --model='ks' \
  --n_lags='2'
```

Results format:

```
{
    "chart.a": {
        "dim.1": {
            "score": 0.0101,
            "score_norm": 0.0063
        },
        "dim.2": {
            "score": 0.0897,
            "score_norm": 0.3652
        }
    },
    "chart.b": {
        "dim.1": {
            "score": 0.3345,
            "score_norm": 0.4444
        },
        "dim.2": {
            "score": 0.6789,
            "score_norm": 0.8907
        },
        "dim.3": {
            "score": 0.0897,
            "score_norm": 0.3652
        }
    }
}
```