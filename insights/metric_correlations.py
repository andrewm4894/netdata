#!/usr/bin/python3
import argparse
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--host', type=str, nargs='?', help='host', default='127.0.0.1'
)
parser.add_argument(
    '--port', type=str, nargs='?', help='port', default='19999'
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
args = parser.parse_args()

# parse args
host = args.host
port = args.port
baseline_after = args.baseline_after
baseline_before = args.baseline_before
highlight_after = args.highlight_after
highlight_before = args.highlight_before

log.info(f"args={args}")
