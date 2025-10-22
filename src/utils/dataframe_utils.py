import datetime

import numpy as np
import pandas as pd

N_BEFORE = 10
N_AFTER = 6


def fetch_past_datetimes(datetime_strs, nlags, timestep):
    try:
        datetimes = pd.to_datetime(datetime_strs)
    except OSError:
        datetimes = pd.to_datetime(np.array(datetime_strs).astype("<U13"))
    relevant_datetimes = [
        [(dt - datetime.timedelta(minutes=timestep * lag)).strftime("%Y%m%d/%H%M") for dt in datetimes]
        for lag in range(nlags)
    ]
    return list(zip(*relevant_datetimes))


def fetch_reversed_past_datetimes(datetime_strs, nlags, timestep):
    try:
        datetimes = pd.to_datetime(datetime_strs)
    except OSError:
        datetimes = pd.to_datetime(np.array(datetime_strs).astype("<U13"))
    relevant_datetimes = [
        [(dt - datetime.timedelta(minutes=timestep * lag)).strftime("%Y%m%d/%H%M") for dt in datetimes]
        for lag in reversed(range(nlags))
    ]
    return list(zip(*relevant_datetimes))


def fetch_future_datetimes(datetime_strs, lags):
    try:
        datetimes = pd.to_datetime(datetime_strs)
    except OSError:
        datetimes = pd.to_datetime(np.array(datetime_strs).astype("<U13"))
    relevant_datetimes = [
        [(dt + datetime.timedelta(minutes=lag)).strftime("%Y%m%d/%H%M") for dt in datetimes] for lag in lags
    ]
    return list(zip(*relevant_datetimes))


def fetch_pred_keys(keys, nlags, timestep):
    try:
        datetimes = pd.to_datetime(keys)
    except OSError:
        datetimes = pd.to_datetime(np.array(keys).astype("<U13"))
    relevant_datetimes = [
        [
            f"{dt.strftime('%Y%m%d/%H%M')}/{(dt + datetime.timedelta(minutes=timestep * lag)).strftime('%Y%m%d-%H%M')}"
            for dt in datetimes
        ]
        for lag in range(1, nlags + 1)
    ]
    return list(zip(*relevant_datetimes))
