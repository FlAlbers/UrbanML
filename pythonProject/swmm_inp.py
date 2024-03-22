#generieren von allen inp files für die simulation von modellregen
# + normale Niederschlagsereignisse
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

# get current path of working directory
current_path = os.getcwd()

###########################################################################################################
# Input section
## Input parameters for inp file generation
base_inp_path = 'pythonProject\\swmm_Gievenbeck.inp'
# path to folder with rain event data
event_data_path = 'pythonProject\\events_FMO'
# path to kostra data
kostra_data_path = os.path.join(current_path, 'pythonProject\\kostra_118111.csv')
# set maximum duration time [min] for Kostra data
max_duration = 72*60
# Name of the study area
name_place = 'Gievenbeck'
# Path to save the inp files
save_inp_path = os.path.join(current_path, 'pythonProject\\inp')
# Euler type for Kostra data (2 is standard)
euler_typ = 2
# Start time of the simulation
start_time = pd.to_datetime('2024-01-01 00:00')
# Buffer time before and after the rainfall event
buffer_time = pd.Timedelta('2h')
# Name of the Kostra time series to be included in the inp file
TSnameKostra = 'Kostra'
# Name of the measured time series to be included in the inp file
TSnameEvent = 'FMO'
# Time interval of the time series in minutes
TSinterval = 5
# amount of cpu cores in the system
cpu_cores = multiprocessing.cpu_count()

# End of input section
############################################################################################################

########################
# read kostra data
kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)
# get return preiods and durations from kostra table
returnrate = kostra.columns.astype(int)
# duration needs to be larger than 15min
durations = kostra.index[(kostra.index >= 15) & (kostra.index <= max_duration)]
# calculate end time of the simulation with start time and buffer time
end_time = start_time + pd.Timedelta(minutes=int(max(durations))) + buffer_time * 2

########################
# read base inp file
inp_base = swmm_api.read_inp_file(base_inp_path)
# Update OPTIONS of inp file
inp_base['OPTIONS'].update({'START_DATE': start_time.date()})
inp_base['OPTIONS'].update({'START_TIME': start_time.time()})
inp_base['OPTIONS'].update({'REPORT_START_DATE': start_time.date()})
inp_base['OPTIONS'].update({'REPORT_START_TIME': start_time.time()})
inp_base['OPTIONS'].update({'END_DATE': end_time.date()})
inp_base['OPTIONS'].update({'END_TIME': end_time.time()})
inp_base['OPTIONS'].update({'THREADS': cpu_cores})

############################################################################################################
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

################################################################################################################
# create inp files
# get all euler model rain series for all return periods and durations
for j in returnrate:
    for d in durations:
        inp = inp_base
        inp['TITLE'] = f'{name_place}_e{euler_typ}_T{int(j)}D{int(d)}' 
        inp = euler_to_inp(inp,kostra, return_period=j, duration=d, interval=5, euler_typ=euler_typ, start_time=start_time + buffer_time, TSname=TSnameKostra)
        for subcatchment in inp['SUBCATCHMENTS']:
            inp['SUBCATCHMENTS'][subcatchment].rain_gage = TSnameKostra        
        inp.write_file(os.path.join(save_inp_path,f'{name_place}_e{euler_typ}_T{int(j)}D{int(d)}.inp'))


del inp[sections.TIMESERIES][TSnameKostra]
del inp['RAINGAGES'][TSnameKostra]
for file_name in os.listdir(event_data_path):
    if file_name.endswith('.csv'):
        
        file_path = os.path.join(event_data_path, file_name)
        event_data = pd.read_csv(file_path)
        inp = event_to_inp(inp, event_data, start_time=start_time + buffer_time, TSname=TSnameEvent)
        for subcatchment in inp['SUBCATCHMENTS']:
            inp['SUBCATCHMENTS'][subcatchment].rain_gage = TSnameEvent
        file_name = file_name.replace('.csv', '')
        file_name = file_name.replace('.', ' ')
        inp['TITLE'] = f'{name_place}_{file_name}'
        inp.write_file(os.path.join(save_inp_path,f'{name_place}_{file_name}.inp'))

# inp = inp_base
# folder_path = 'pythonProject\\events_FMO'      
# file_path = os.path.join(folder_path, '2014-02-01 07 20 00_hN1.56.csv')
# event_data = pd.read_csv(file_path)
# for subcatchment in inp['SUBCATCHMENTS']:
#     inp['SUBCATCHMENTS'][subcatchment].rain_gage = TSnameEvent
# inp = event_to_inp(inp_base, event_data, start_time=start_time + buffer_time, TSname=TSnameEvent)

# inp.write_file(os.path.join(save_inp_path,'2014-02-01 07 20 00_hN1.56.inp'))

# event_data = event_data.set_index('date')
# TimeseriesData(TSnameEvent, data=list(zip(event_data.index, event_data['precipitation_height'])))
# inp[sections.TIMESERIES][TSnameEvent] = TimeseriesData(TSnameEvent, data=list(zip(event_data.index, event_data['precipitation_height'])))

# inp[sections.TIMESERIES][TSnameEvent]
# inp_base['TITLE']
# gievenbeck = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')