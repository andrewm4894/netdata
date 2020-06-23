## Insights

Statistics and machine learning based "Insights" features.

### metric_correlations

`metric_correlations.py` is a script to take in some comand line args, pull data from netdata api, calculate scores per dimension and return a json result.

#### Example usage:

```
# showing main params:
python metric_correlations.py \
  --host='127.0.0.1:19999' \
  --baseline_after='-480' \
  --baseline_before='-240' \
  --highlight_after='-240' \
  --highlight_before='0' \
  --model='hbos' \
  --n_lags='2' \
  --model_level='chart'

# typical usage:
python metric_correlations.py --baseline_after='-480' --baseline_before='-240' --highlight_after='-240' --highlight_before='0' --model='ks'
```

#### Results format:

```
# for a model_level='dim' example (where have scores at the individual dimension level)
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

# for a model_level='chart' example (where we only have scores at the overall chart level)
{
    "chart.a": {
        "*": {
            "score": 0.0101,
            "score_norm": 0.0063
        }
    },
    "chart.b": {
        "*": {
            "score": 0.3345,
            "score_norm": 0.4444
        }
    }
}
```

#### Benchmarks:

`run_benchmarks.py` is a little python script that loops over a list of models and sample sizes and prints some timings, split by time spent getting the data and then time spent running the models.

```
# example usage:
python run_benchmarks.py --model_list='ks,knn,hbos' --n_list='100,1000,5000,10000' --model_level='dim'

# example results:
   model level  success  default  fail    t_n  t_data  t_scores  t_total
0     ks   dim      335        0     0    100    2.86      0.05     2.91
1     ks   dim      271        0     0   1000    4.15      0.07     4.22
2     ks   dim      224        0     0   5000   11.82      0.16    11.98
3     ks   dim      191        0     0  10000   32.70      0.25    32.95
4    knn   dim      339        0     0    100    3.09      4.24     7.33
5    knn   dim      271        0     0   1000    4.55     22.53    27.07
6    knn   dim      224        0     0   5000   13.54     93.77   107.31
7    knn   dim      191        0     0  10000   26.32    170.13   196.45
8   hbos   dim      339        0     0    100    2.76      3.88     6.64
9   hbos   dim      271        0     0   1000    4.70      1.94     6.63
10  hbos   dim      224        0     0   5000   14.97      0.81    15.78
11  hbos   dim      191        0     0  10000   31.06      0.88    31.95
```

#### Tests:

`test_metric_correlations.py` contains some tests, use `pytest` to run them.
