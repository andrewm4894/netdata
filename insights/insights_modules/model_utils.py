import logging

log = logging.getLogger(__name__)


def try_fit(clf, colname, data, default_model):
    try:
        clf.fit(data)
        result = 'success'
    except Exception as e:
        log.warning(e)
        log.info(f'... could not fit model for {colname}, trying default')
        clf = default_model()
        clf.fit(data)
        result = 'default'
    return clf, result


def init_counters(colnames):
    n_charts, n_dims = len(set([colname.split('|')[0] for colname in colnames])), len(colnames)
    n_bad_data, fit_success, fit_default, fit_fail = 0, 0, 0, 0
    return n_charts, n_dims, n_bad_data, fit_success, fit_default, fit_fail


def get_col_map(colnames, model_level):
    col_map = {}
    if model_level == 'chart':
        charts_list = list(set([colname.split('|')[0] for colname in colnames]))
        for chart in charts_list:
            col_map[chart] = [colnames.index(colname) for colname in colnames if colname.startswith(f'{chart}|')]
    else:
        for col in colnames:
            col_map[col] = [colnames.index(colname) for colname in colnames if colname == col]
    return col_map


def summary_info(n_charts, n_dims, n_bad_data, fit_success, fit_fail, fit_default, model_level='dim'):
    # log some summary stats
    if model_level == 'chart':
        success_rate = round(fit_success / n_charts, 2)
        bad_data_rate = round(n_bad_data / n_charts, 2)
    else:
        bad_data_rate = round(n_bad_data / n_dims, 2)
        success_rate = round(fit_success / n_dims, 2)
    msg = f"... success_rate={success_rate}, bad_data_rate={bad_data_rate}, charts={n_charts}, dims={n_dims}"
    msg += f", bad_data={n_bad_data}, fit_success={fit_success}, fit_fail={fit_fail}, fit_default={fit_default}"
    return msg