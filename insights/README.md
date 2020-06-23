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
python run_benchmarks.py --model_list='ks,knn,hbos' --n_list='100,1000,5000,10000' --model_level='chart'

# example results:
   model      n  time_data  time_scores  time_total
0     ks    100       3.08         0.36        3.44
1     ks   1000       5.32         2.47        7.79
2     ks   5000      14.89         0.18       15.06
3     ks  10000      35.86         0.27       36.13
4    knn    100       3.05         4.41        7.46
5    knn   1000       5.60        23.45       29.06
6    knn   5000      15.82        68.15       83.97
7    knn  10000      31.48       144.43      175.91
8   hbos    100       3.55         4.02        7.57
9   hbos   1000       5.06         2.21        7.27
10  hbos   5000      15.66         0.64       16.30
11  hbos  10000      43.85         0.79       44.63
 
```