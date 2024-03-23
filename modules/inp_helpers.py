import os
from io import StringIO
import pandas as pd
from pandas import read_csv
from ehyd_tools.synthetic_rainseries import RainModeller

import swmm_api
from swmm_api import SwmmInput
from swmm_api.input_file import section_labels as sections
from swmm_api.input_file.section_types import SECTION_TYPES
from swmm_api.input_file.sections.others import RainGage

from swmm_api.input_file.sections import Timeseries
from swmm_api.input_file.sections.others import TimeseriesData, TimeseriesFile
import multiprocessing

# helper functions
# helper function to get euler model rain series from kostra table
def get_euler_ts(kostra_data, return_period, duration, interval = 5, euler_typ = 2, start_time = '2024-01-01 00:00'):
    # kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)
    model_rain = RainModeller()
    model_rain.idf_table = kostra_data
    model_rain.idf_table.columns = model_rain.idf_table.columns.astype(int)
    ts = model_rain.euler.get_time_series(return_period=int(return_period), duration=int(duration), interval=int(interval), kind = int(euler_typ), start_time=start_time)
    ts = ts.round(2)
    ts = ts.rename('KOSTRA')
    return ts

# helper function to add euler model rain series to inp file
def euler_to_inp(SWMM_inp,kostra_data, return_period, duration, interval = 5, euler_typ = None, start_time = '',TSname = 'Kostra'):
    euler2 = get_euler_ts(kostra_data, return_period=return_period, duration=duration, interval=interval, euler_typ=euler_typ, start_time=start_time)
    # Convert TSinterval to swmm format 
    if interval >= 10:
        TSinterval_time = f'0:{interval}'
    else:
        TSinterval_time = f'0:0{interval}'
    euler2.name = TSname
    # input[sections.TIMESERIES].add_obj(TimeseriesData(name, data=list(zip(euler2.index,euler2))))
    SWMM_inp[sections.TIMESERIES][TSname] =  TimeseriesData(TSname, data=list(zip(euler2.index,euler2)))
    SWMM_inp['RAINGAGES'][TSname] = RainGage(TSname, 'VOLUME', TSinterval_time, 1 ,'TIMESERIES', TSname)
    return SWMM_inp

# helper function to add measured rain series to inp file
def event_to_inp(SWMM_inp, event_data, start_time='2024-01-01 00:00', interval=5 , TSname='Event'):
    event_data['date'] = pd.to_datetime(event_data['date'])
    event_data['date'] = event_data['date'].apply(lambda x: start_time + pd.Timedelta(minutes=(x - event_data['date'].iloc[0]).total_seconds() / 60))
    SWMM_inp[sections.TIMESERIES][TSname] = TimeseriesData(TSname, data=list(zip(event_data['date'], event_data['precipitation_height'])))
    #Convert TSinterval to swmm format 
    if interval >= 10:
        TSinterval_time = f'0:{interval}'
    else:
        TSinterval_time = f'0:0{interval}'
    SWMM_inp['RAINGAGES'][TSname] = RainGage(TSname, 'VOLUME', TSinterval_time, 1 ,'TIMESERIES', TSname)
    return SWMM_inp