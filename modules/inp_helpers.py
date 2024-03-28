"""
Author: Flemming Albers

This script contains helper functions for working with SWMM input files and rain series data.

The helper functions in this script are as follows:
- get_euler_ts: This function returns an Euler model rain series by return period and duration based on a given kostra table.
- euler_to_inp: This function adds Euler model rain series to the SWMM input file.
- event_to_inp: This function adds the measured rain series to the SWMM input file.

"""
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
def get_euler_ts(kostra_data, return_period, duration, interval=5, euler_typ=2, start_time='2024-01-01 00:00'):
    """
    Generate a time series using the Euler method based on KOSTRA data.

    Parameters:
    - kostra_data (DataFrame): The KOSTRA data used for modeling.
    - return_period (int): The return period in years.
    - duration (int): The duration of the event in minutes.
    - interval (int, optional): The time interval between data points in minutes. Default is 5 minutes.
    - euler_typ (int, optional): The type of Euler method to use. Default is 2.
    - start_time (str, optional): The start time of the time series in 'YYYY-MM-DD HH:MM' format. Default is '2024-01-01 00:00'.

    Returns:
        pd.Series: The generated time series based on the Euler method.
    """
    model_rain = RainModeller()
    model_rain.idf_table = kostra_data
    model_rain.idf_table.columns = model_rain.idf_table.columns.astype(int)
    ts = model_rain.euler.get_time_series(return_period=int(return_period), duration=int(duration), interval=int(interval), kind=int(euler_typ), start_time=start_time)
    ts = ts.round(2)
    ts = ts.rename('KOSTRA')
    return ts

# helper function to add euler model rain series to inp file
def euler_to_inp(SWMM_inp, kostra_data, return_period, duration, interval=5, euler_typ=2, start_time='2024-01-01 00:00', TSname='Kostra', buffer_time=pd.Timedelta('2h')):
    """
    Converts Euler data to SWMM input format and updates the SWMM input file.

    Parameters:
    - SWMM_inp (dict): The SWMM input file as a dictionary.
    - kostra_data (pd.DataFrame): The Euler data.
    - return_period (int): The return period of the event.
    - duration (int): The duration of the event.
    - interval (int, optional): The time interval of the event steps. Defaults to 5.
    - euler_typ (str, optional): The type of Euler data. 1 or 2 possible as input. Defaults to 2.
    - start_time (str, optional): The output start time of the event. Defaults to '2024-01-01 00:00'. Only important for SWMM inp file.
    - TSname (str, optional): The name of the timeseries in SWMM inp file. Defaults to 'Kostra'.
    - buffer_time (pd.Timedelta, optional): The buffer time to add before and after the event. Defaults to pd.Timedelta('2h'). Only important for SWMM inp file.

    Returns:
    - SWMM_inp (dict): The updated SWMM input file with the kostra-event data.
    """
    start_time = start_time + buffer_time
    euler2 = get_euler_ts(kostra_data, return_period=return_period, duration=duration, interval=interval, euler_typ=euler_typ, start_time=start_time)
    # Convert TSinterval to swmm format
    if interval >= 10:
        TSinterval_time = f'0:{interval}'
    else:
        TSinterval_time = f'0:0{interval}'
    euler2.name = TSname
    SWMM_inp[sections.TIMESERIES][TSname] = TimeseriesData(TSname, data=list(zip(euler2.index, euler2)))
    SWMM_inp['RAINGAGES'][TSname] = RainGage(TSname, 'VOLUME', TSinterval_time, 1, 'TIMESERIES', TSname)
    end_time = start_time + pd.Timedelta(minutes=int(duration)) + buffer_time * 2
    SWMM_inp['OPTIONS'].update({'END_DATE': end_time.date()})
    SWMM_inp['OPTIONS'].update({'END_TIME': end_time.time()})
    return SWMM_inp

# helper function to add measured rain series to inp file
def event_to_inp(SWMM_inp, event_data, start_time='2024-01-01 00:00', interval=5 , TSname='Event', buffer_time = pd.Timedelta('2h')):
    """
    Converts event data into SWMM input format and updates the SWMM input file.

    Parameters:
    - SWMM_inp (dict): The SWMM input file as a dictionary.
    - event_data (DataFrame): The event data containing precipitation information.    
    - start_time (str, optional): The output start time of the event. Defaults to '2024-01-01 00:00'. Only important for SWMM inp file.
    - interval (int, optional): The time interval of the event steps. Defaults to 5.
    - TSname (str, optional): The name of the timeseries in SWMM inp file. Defaults to 'Kostra'.
    - buffer_time (pd.Timedelta, optional): The buffer time to add before and after the event. Defaults to pd.Timedelta('2h'). Only important for SWMM inp file.

    Returns:
    - SWMM_inp (dict): The updated SWMM input file with the event data.

    """

    event_data['date'] = pd.to_datetime(event_data['date'])
    event_data['date'] = event_data['date'].apply(lambda x: start_time + buffer_time + pd.Timedelta(minutes=(x - event_data['date'].iloc[0]).total_seconds() / 60))
    SWMM_inp[sections.TIMESERIES][TSname] = TimeseriesData(TSname, data=list(zip(event_data['date'], event_data['precipitation_height'])))
    #Convert TSinterval to swmm format 
    if interval >= 10:
        TSinterval_time = f'0:{interval}'
    else:
        TSinterval_time = f'0:0{interval}'
    
    end_time = max(event_data['date']) + buffer_time
    SWMM_inp['OPTIONS'].update({'END_DATE': end_time.date()})
    SWMM_inp['OPTIONS'].update({'END_TIME': end_time.time()})
    SWMM_inp['RAINGAGES'][TSname] = RainGage(TSname, 'VOLUME', TSinterval_time, 1 ,'TIMESERIES', TSname)
    return SWMM_inp