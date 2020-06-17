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

Example response:

```
{
    "groups.lreads": {
        "unscd": {
            "score": 0.0101,
            "score_norm": 0.0063
        },
        "www-data": {
            "score": 0.0897,
            "score_norm": 0.3652
        },
        "netdata": {
            "score": 0.0972,
            "score_norm": 0.399
        },
        "root": {
            "score": 0.1115,
            "score_norm": 0.4635
        }
    },
    "groups.cpu_system": {
        "mysql": {
            "score": 0.0619,
            "score_norm": 0.2399
        },
        "www-data": {
            "score": 0.0409,
            "score_norm": 0.1452
        },
        "netdata": {
            "score": 0.0619,
            "score_norm": 0.2399
        },
        "root": {
            "score": 0.0615,
            "score_norm": 0.2381
        }
    },
    "groups.pwrites": {
        "www-data": {
            "score": 0.0443,
            "score_norm": 0.1605
        },
        "netdata": {
            "score": 0.0722,
            "score_norm": 0.2863
        },
        "root": {
            "score": 0.043,
            "score_norm": 0.1546
        }
    },
    "groups.vmem": {
        "postdrop": {
            "score": 0.1265,
            "score_norm": 0.5311
        }
    },
    "groups.mem": {
        "mysql": {
            "score": 0.1017,
            "score_norm": 0.4193
        },
        "postdrop": {
            "score": 0.1347,
            "score_norm": 0.5681
        }
    },
    "groups.minor_faults": {
        "mysql": {
            "score": 0.0699,
            "score_norm": 0.2759
        },
        "netdata": {
            "score": 0.0739,
            "score_norm": 0.294
        },
        "root": {
            "score": 0.0758,
            "score_norm": 0.3025
        }
    },
    "groups.cpu": {
        "mysql": {
            "score": 0.0523,
            "score_norm": 0.1966
        },
        "www-data": {
            "score": 0.0404,
            "score_norm": 0.1429
        },
        "netdata": {
            "score": 0.08,
            "score_norm": 0.3215
        },
        "root": {
            "score": 0.0777,
            "score_norm": 0.3111
        }
    },
    "users.sockets": {
        "www-data": {
            "score": 0.0549,
            "score_norm": 0.2083
        }
    },
    ...
}
```