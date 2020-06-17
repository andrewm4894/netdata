import argparse
import logging
import time

from netdata_pandas.data import get_data, get_chart_list
from insights_modules.model import run_model


time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--host', type=str, nargs='?', help='host', default='127.0.0.1:19999'
)
parser.add_argument(
    '--baseline_after', type=str, nargs='?', help='baseline_after', default='-120'
)
parser.add_argument(
    '--baseline_before', type=str, nargs='?', help='baseline_before', default='-60'
)
parser.add_argument(
    '--highlight_after', type=str, nargs='?', help='highlight_after', default='-60'
)
parser.add_argument(
    '--highlight_before', type=str, nargs='?', help='highlight_before', default='0'
)
parser.add_argument(
    '--model', type=str, nargs='?', help='model', default='ks'
)
parser.add_argument(
    '--n_lags', type=str, nargs='?', help='n_lags', default='2'
)
parser.add_argument(
    '--log_level', type=str, nargs='?', help='log_level', default='info'
)
args = parser.parse_args()

# parse args
host = args.host
baseline_after = int(args.baseline_after)
baseline_before = int(args.baseline_before)
highlight_after = int(args.highlight_after)
highlight_before = int(args.highlight_before)
model = args.model
n_lags = args.n_lags
log_level = args.log_level

# handle 'after' and 'before' values if passed in as relative
now = time.time()
if baseline_after <= 0:
    baseline_after = int(now + baseline_after)
if baseline_before <= 0:
    baseline_before = int(now + baseline_before)
if highlight_after <= 0:
    highlight_after = int(now + highlight_after)
if highlight_before <= 0:
    highlight_before = int(now + highlight_before)

# set up logging
if log_level == 'info':
    logging.basicConfig(level=logging.INFO)
elif log_level == 'debug':
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARN)

log = logging.getLogger(__name__)

log.info(f"... args={args}")

log.debug(f"... baseline_after={baseline_after}")
log.debug(f"... baseline_before={baseline_before}")
log.debug(f"... highlight_after={highlight_after}")
log.debug(f"... highlight_before={highlight_before}")

# get charts
charts = get_chart_list(host)

# get data
df = get_data(host, charts, after=baseline_after, before=highlight_before, diff=True,
              ffill=True, numeric_only=True, nunique_thold=0.05, col_sep='|')

log.info(f"... df.shape={df.shape}")

# get numpy arrays
colnames = list(df.columns)
arr_baseline = df.query(f'{baseline_after} <= time_idx <= {baseline_before}').values
arr_highlight = df.query(f'{highlight_after} <= time_idx <= {highlight_before}').values
charts = list(set([col.split('|')[0] for col in colnames]))

log.debug(f'... arr_baseline.shape = {arr_baseline.shape}')
log.debug(f'... arr_highlight.shape = {arr_highlight.shape}')

# log times
time_got_data = time.time()

log.info(f'... {round(time_got_data - time_start,2)} seconds to get data.')

# get scores
results = run_model(model, colnames, arr_baseline, arr_highlight, n_lags)

print(results)


