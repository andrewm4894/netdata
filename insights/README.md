## Insights

Location for some Machine Learning and Statistical based features.

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