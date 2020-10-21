<!--
---
title: "anomalies_online"
custom_edit_url: https://github.com/netdata/netdata/edit/master/collectors/python.d.plugin/anomalies_online/README.md
---
-->

# anomalies_online

foo. 

## Charts

One chart is produced:

- **Anomaly Score** (`anomalies.score`): xxx.

xxx.

## Requirements

- This collector will only work with Python 3 and requires the packages below be installed.

```bash
# become netdata user
sudo su -s /bin/bash netdata
# install required packages for the netdata user
pip3 install --user netdata-pandas==0.0.23 numba==0.50.1 mmh3==2.5.1 rrcf==0.4.3 pysad==0.1.1
```

## Configuration

Install the Python requirements above, enable the collector and restart Netdata.

```bash
cd /etc/netdata/
sudo ./edit-config python.d.conf
# Set `anomalies_online: no` to `anomalies_online: yes`
sudo service netdata restart
```

xxx.

## Custom Models

In the `anomalies.conf` file you can also define some "custom models" which you can use to group one or more metrics into a single model much like is done by default for the charts you specify. This is useful if you have a handful of metrics that exist in different charts but perhaps are related to the same underlying thing you would like to perform anomaly detection on, for example a specific app or user. 

To define a custom model you would include configuation like below in `anomalies.conf`. By default there should already be some commented out examples in there. 

`name` is a name you give your custom model, this is what will appear alongside any other specified charts in the `anomalies.probability` and `anomalies.anomaly` charts. `dimensions` is a string of metrics you want to include in your custom model. By default the [netdata-pandas](https://github.com/netdata/netdata-pandas) library used to pull the data from Netdata uses a "chart.a|dim.1" type of naming convention in the pandas columns it returns, hence the `dimensions` string should look like "chart.name|dimension.name,chart.name|dimension.name". The examples below hopefully make this clear.

```yaml
custom_models:
 - name: 'user_netdata'
   dimensions: 'users.cpu|netdata,users.mem|netdata,users.threads|netdata,users.processes|netdata,users.sockets|netdata'
 - name: 'apps_python_d_plugin'
   dimensions: 'apps.cpu|python.d.plugin,apps.mem|python.d.plugin,apps.threads|python.d.plugin,apps.processes|python.d.plugin,apps.sockets|python.d.plugin'
custom_models_normalize: false
```

## Troubleshooting

To see any relevant log messages you can use a command like `grep 'anomalies_online' /var/log/netdata/error.log`.

If you would like to log in as `netdata` user and run the collector in debug mode to see more detail.

```bash
# become netdata user
sudo su -s /bin/bash netdata
# run collector in debug using `nolock` option if netdata is already running the collector itself.
/usr/libexec/netdata/plugins.d/python.d.plugin anomalies_online debug trace nolock
```

## Notes

- xxx.

## Useful Links & Further Reading

- xxx.