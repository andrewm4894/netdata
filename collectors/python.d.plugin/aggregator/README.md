<!--
title: "aggregator"
custom_edit_url: https://github.com/netdata/netdata/edit/master/collectors/python.d.plugin/aggregator/README.md
-->

## Aggregator

This collector 'aggregates' charts from multiple children that are streaming to a parent node. 

### Charts

You should see charts similar to those you have configured to aggregate. For example in the below chart underneath the "Aggregator devml" context we see the "system cpu" chart with is just the aggregation of the system.cpu chart over the specified children nodes. 

![netdata-aggregator-collector](https://github.com/andrewm4894/random/blob/master/images/netdata/netdata-aggregator-collector.jpg)

### Requirements

### Configuration

### Troubleshooting

### Notes

- xxx

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
sudo systemctl restart netdata
```

```
sudo su -s /bin/bash netdata
/usr/libexec/netdata/plugins.d/python.d.plugin aggregator debug trace nolock 

```