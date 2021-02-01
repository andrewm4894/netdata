<!--
title: "aggregator"
custom_edit_url: https://github.com/netdata/netdata/edit/master/collectors/python.d.plugin/aggregator/README.md
-->

## Aggregator

todo

```bash
# become netdata user
sudo su -s /bin/bash netdata
# install required packages for the netdata user
pip3 install --user numpy==1.19.5 requests==2.25.1
```

```
grep 'aggregator' /var/log/netdata/error.log
```

```
cd netdata
git pull
sudo cp collectors/python.d.plugin/aggregator/aggregator.chart.py /usr/libexec/netdata/python.d/
```