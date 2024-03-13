#generieren von allen inp files für die simulation von modellregen
# + normale Niederschlagsereignisse
# hinterlegen an alle flächen
import os
from io import StringIO
import pandas as pd
from pandas import read_csv
from ehyd_tools.synthetic_rainseries import RainModeller

import swmm_api
from swmm_api import SwmmInput
from swmm_api.input_file import section_labels as sections
from swmm_api.input_file.section_types import SECTION_TYPES

from swmm_api.input_file.sections import Timeseries
from swmm_api.input_file.sections.others import TimeseriesData, TimeseriesFile


# get current path of working directory
current_path = os.getcwd()
# path to kostra data
kostra_data_path = os.path.join(current_path, 'pythonProject\\kostra_118111.csv')
# read kostra data
kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)


# helper function to get euler model rain series from kostra table
def get_euler_ts(kostra, return_period, duration, interval = 5, euler_typ = 2, start_time = '2024-01-01 00:00'):
    # kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)
    model_rain = RainModeller()
    model_rain.idf_table = kostra
    model_rain.idf_table.columns = model_rain.idf_table.columns.astype(int)
    ts = model_rain.euler.get_time_series(return_period=int(return_period), duration=int(duration), interval=int(interval), kind = int(euler_typ), start_time=start_time)
    ts = ts.round(2)
    ts = ts.rename('KOSTRA')
    return ts

# helper function to add euler model rain series to inp file
def euler_to_inp(input,kostra, return_period, duration, interval = None, euler_typ = None, start_time = '',TSname = 'Kostra'):
    euler2 = get_euler_ts(kostra, return_period=return_period, duration=duration, interval=interval, euler_typ=euler_typ, start_time=start_time)
    # name = f'e2_T{int(return_period)}D{int(duration)}'
    euler2.name = TSname
    # input[sections.TIMESERIES].add_obj(TimeseriesData(name, data=list(zip(euler2.index,euler2))))
    input[sections.TIMESERIES][TSname] =  TimeseriesData(TSname, data=list(zip(euler2.index,euler2)))
    return input

## Input parameters for inp file generation
# get return preiods and durations from kostra table
jaerlichkeiten = kostra.columns.astype(int)
# duration needs to be larger than 15min
dauern = kostra.index[kostra.index >= 15]
name_place = 'Gievenbeck'
save_inp_path = os.path.join(current_path, 'pythonProject\\inp')
euler_typ = 2
start_time = pd.to_datetime('2024-01-01 00:00')
buffer_time = pd.Timedelta('2h')
TSname = 'Kostra'
TSinterval = 5



#Convert TSinterval to swmm format 
if TSinterval >= 10:
    TSinterval_time = f'0:{TSinterval}'
else:
    TSinterval_time = f'0:0{TSinterval}'

# read inp file
inp_base = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')

# Update OPTIONS of inp file
inp_base['OPTIONS'].update({'START_DATE': start_time.date()})
inp_base['OPTIONS'].update({'START_TIME': start_time.time()})
inp_base['OPTIONS'].update({'REPORT_START_DATE': start_time.date()})
inp_base['OPTIONS'].update({'REPORT_START_TIME': start_time.time()})
inp_base['RAINGAGES']['Kostra']['form'] = 'VOLUME'
inp_base['RAINGAGES']['Kostra']['interval'] = TSinterval_time 
inp_base['RAINGAGES']['Kostra']['source'] = 'TIMESERIES'
inp_base['RAINGAGES']['Kostra']['timeseries'] = TSname

# type(inp_base[sections.TIMESERIES])
# inp_base[sections.TIMESERIES]['Kostra']
# inp_base[sections.TIMESERIES]['e2_T2D120']

# inp_base = euler_to_inp(inp_base,kostra, return_period=2, duration=120, interval=5, euler_typ=euler_typ, start_time=start_time + buffer_time)


#get all euler model rain series for all return periods and durations
for j in jaerlichkeiten:
    for d in dauern:
        inp = inp_base
        end_time = start_time + pd.Timedelta(minutes=int(d)) + buffer_time * 2
        inp = euler_to_inp(inp,kostra, return_period=j, duration=d, interval=5, euler_typ=euler_typ, start_time=start_time + buffer_time)
        for subcatchment in inp['SUBCATCHMENTS']:
            inp['SUBCATCHMENTS'][subcatchment].rain_gage = TSname
        inp['OPTIONS'].update({'END_DATE': end_time.date()})
        inp['OPTIONS'].update({'END_TIME': end_time.time()})
        # inp[sections.TIMESERIES].update(inp_base[sections.TIMESERIES])
        inp.write_file(os.path.join(save_inp_path,f'{name_place}_e{euler_typ}_T{int(j)}D{int(d)}.inp'))
        # inp = SwmmInput()








# gievenbeck = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')
