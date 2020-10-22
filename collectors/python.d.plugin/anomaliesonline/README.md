<!--
---
title: "anomaliesonline"
custom_edit_url: https://github.com/netdata/netdata/edit/master/collectors/python.d.plugin/anomaliesonline/README.md
---
-->

# Anomlaies Online - online/streaming anomaly detection for your Netdata!

This collector uses the Python [PySAD](https://pysad.readthedocs.io/en/latest/) library to perform [online](https://en.wikipedia.org/wiki/Online_machine_learning) unsupervised [anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection) on your Netdata charts and/or dimensions.

Instead of this collector just _collecting_ data, it also does some computation on the data it collects to return an anomaly probability and anomaly flag for each chart or custom model you define. This computation consists of a **predict** function that continually fits and predicts anomaly probabilities in an online/streaming manner (e.g. there is no relativley expensive training step as all the models used are '[online](https://en.wikipedia.org/wiki/Online_machine_learning)' models). 

## Charts

Two charts are produced:

- **Anomaly Probability** (`anomaliesonline.probability`): This chart shows the probability that the latest observed data is anomalous based on the most recent streams of data seen by the model for that chart (using the [`fit_score_partial()`](https://pysad.readthedocs.io/en/latest/generated/pysad.core.BaseModel.html#pysad.core.BaseModel.fit_score_partial) method of the PySAD model).
- **Anomaly** (`anomaliesonline.anomaly`): This chart shows `1` or `0` predictions of if the latest observed data is considered anomalous or not based on the fitted model (using the [`fit_transform_partial()`](https://pysad.readthedocs.io/en/latest/generated/pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator.html#pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator.fit_transform_partial) transformer from PySAD.

Below is an example of the charts produced by this collector and how they might look when things are 'normal' on the node. The anomaly probabilities tend to bounce randomly around a typical probability range, one or two might randomly jump or drift outside of this range every now and then and show up as anomalies on the anomaly chart. 

![alt text](https://github.com/andrewm4894/random/blob/master/images/netdata/netdata-anomalies-online-collector-normal.jpg)

If we then go onto the system and run a command like `stress-ng --all 2` to create some [stress](https://wiki.ubuntu.com/Kernel/Reference/stress-ng), we see some charts begin to have anomaly probabilities that jump outside the typical range. When the anomaly probabilities change enough, we will start seeing anomalies being flagged on the `anomaliesonline.anomaly` chart. The idea is that these charts are the most anomalous right now so could be a good place to start your troubleshooting. 

![alt text](https://github.com/andrewm4894/random/blob/master/images/netdata/netdata-anomalies-online-collector-abnormal.jpg)

Then, as the issue passes, the anomaly probabilities should settle back down into their 'normal' range again. 

![alt text](https://github.com/andrewm4894/random/blob/master/images/netdata/netdata-anomalies-online-collector-normal-again.jpg)

## Requirements

- This collector will only work with Python 3 and requires the packages below be installed.

```bash
# become netdata user
sudo su -s /bin/bash netdata
# install required packages for the netdata user
pip3 install --user netdata-pandas==0.0.24 numba==0.50.1 mmh3==2.5.1 rrcf==0.4.3 pysad==0.1.1
```

## Configuration

Install the Python requirements above, enable the collector and restart Netdata.

```bash
cd /etc/netdata/
sudo ./edit-config python.d.conf
# Set `anomaliesonline: no` to `anomaliesonline: yes`
sudo service netdata restart
```

The configuration for the anomalies collector defines how it will behave on your system and might take some experimentation with over time to set it optimally for your node. Out of the box, the config comes with some [sane defaults](https://www.netdata.cloud/blog/redefining-monitoring-netdata/) to get you started that try to balance the flexibility and power of the ML models with the goal of being as cheap as possible in term of cost on the node resources. 

_**Note**: If you are unsure about any of the below configuration options then it's best to just ignore all this and leave the `anomalies.conf` file alone to begin with. Then you can return to it later if you would like to tune things a bit more once the collector is running for a while and you have a feeling for its performance on your node._

Edit the `python.d/anomaliesonline.conf` configuration file using `edit-config` from the your agent's [config
directory](https://learn.netdata.cloud/guides/step-by-step/step-04#find-your-netdataconf-file), which is usually at `/etc/netdata`.

```bash
cd /etc/netdata   # Replace this path with your Netdata config directory, if different
sudo ./edit-config python.d/anomaliesonline.conf
```

The default configuration should look something like this. Here you can see each parameter (with sane defaults) and some information about each one and what it does.

```yaml
# use http or https to pull data
protocol: 'http'
# what host to pull data from.
host: '127.0.0.1:19999'
# what charts to pull data for - A regex like 'system\..*|' or 'system\..*|apps.cpu|apps.mem' etc.
charts_in_scope: 'system\..*'
# what model to use - can be one of 'rrcf'
model: 'rrcf'
# how many lagged values of each dimension to include in the 'feature vector' each model is fit on.
lags_n: 3
# how much smoothing to apply to each dimension in the 'feature vector' each model is fit on.
smooth_n: 3
# how many differences to take in preprocessing your data. diffs_n=0 would mean fitting models on the raw values of each dimension, whereas diffs_n=1 means everything is done in terms of differences.
diffs_n: 1
# The size of the rolling window you want to use when converting a raw score from pysad into a probability using pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator.
calibrator_window_size: 1000
# The size of the window used in a running average over the anomaly probabilities produced by pysad using pysad.transform.postprocessing.RunningAveragePostprocessor.
postprocessor_window_size: 15
# threshold over which you want to trigger an anomaly flag on the anomaly chart.
anomaly_threshold: 90

# define any custom models you would like to create anomaly probabilties for, some examples below to show how.
# for example below example creates two custom models, one to run anomaly detection on the netdata user and one on the apps metrics for python.d.plugin.
custom_models:
 - name: 'user_netdata'
   dimensions: 'users.cpu|netdata,users.mem|netdata,users.threads|netdata,users.processes|netdata,users.sockets|netdata'
 - name: 'apps_python_d_plugin'
   dimensions: 'apps.cpu|python.d.plugin,apps.mem|python.d.plugin,apps.threads|python.d.plugin,apps.processes|python.d.plugin,apps.sockets|python.d.plugin'
```

## Custom Models

In the `anomaliesonline.conf` file you can also define some "custom models" which you can use to group one or more metrics into a single model much like is done by default for the charts you specify. This is useful if you have a handful of metrics that exist in different charts but perhaps are related to the same underlying thing you would like to perform anomaly detection on, for example a specific app or user. 

To define a custom model you would include configuation like below in `anomaliesonline.conf`. By default there should already be some commented out examples in there. 

`name` is a name you give your custom model, this is what will appear alongside any other specified charts in the `anomaliesonline.probability` and `anomaliesonline.anomaly` charts. `dimensions` is a string of metrics you want to include in your custom model. By default the [netdata-pandas](https://github.com/netdata/netdata-pandas) library used to pull the data from Netdata uses a "chart.a|dim.1" type of naming convention in the pandas columns it returns, hence the `dimensions` string should look like "chart.name|dimension.name,chart.name|dimension.name". The examples below hopefully make this clear.

```yaml
custom_models:
 - name: 'user_netdata'
   dimensions: 'users.cpu|netdata,users.mem|netdata,users.threads|netdata,users.processes|netdata,users.sockets|netdata'
 - name: 'apps_python_d_plugin'
   dimensions: 'apps.cpu|python.d.plugin,apps.mem|python.d.plugin,apps.threads|python.d.plugin,apps.processes|python.d.plugin,apps.sockets|python.d.plugin'
custom_models_normalize: false
```

## Troubleshooting

To see any relevant log messages you can use a command like `grep 'anomaliesonline' /var/log/netdata/error.log`.

If you would like to log in as `netdata` user and run the collector in debug mode to see more detail.

```bash
# become netdata user
sudo su -s /bin/bash netdata
# run collector in debug using `nolock` option if netdata is already running the collector itself.
/usr/libexec/netdata/plugins.d/python.d.plugin anomaliesonline debug trace nolock
```

## Notes

- xxx.

## Useful Links & Further Reading

- [PySAD documentation](https://pysad.readthedocs.io/en/latest/index.html), [PySAD GitHub](https://github.com/selimfirat/pysad).