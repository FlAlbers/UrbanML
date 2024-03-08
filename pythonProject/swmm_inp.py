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

#get current path of working directory
current_path = os.getcwd()
#path to kostra data
kostra_data_path = os.path.join(current_path, 'pythonProject\\kostra_118111.csv')
#read kostra data
kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)


#helper function to get euler model rain series from kostra table
def get_euler_ts(kostra, return_period, duration, interval = 5, euler_typ = 2, start_time = '2024-01-01 00:00'):
    # kostra = pd.read_csv(kostra_data_path, delimiter=',', index_col=0)
    model_rain = RainModeller()
    model_rain.idf_table = kostra
    model_rain.idf_table.columns = model_rain.idf_table.columns.astype(int)
    ts = model_rain.euler.get_time_series(return_period=int(return_period), duration=int(duration), interval=int(interval), kind = int(euler_typ), start_time=start_time)
    ts = ts.round(2)
    ts = ts.rename('KOSTRA')
    return ts

#helper function to add euler model rain series to inp file
def euler_to_inp(inp,kostra, return_period, duration, interval = None, euler_typ = None, start_time = ''):
    euler2 = get_euler_ts(kostra, return_period=return_period, duration=duration, interval=interval, euler_typ=euler_typ, start_time=start_time)
    name = f'e2_T{int(return_period)}D{int(duration)}'
    euler2.name = name
    inp[sections.TIMESERIES].add_obj(TimeseriesData(name, data=list(zip(euler2.index,euler2))))
    return inp

#read inp file
inp_base = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')

#get return preiods and durations from kostra table
jaerlichkeiten = kostra.columns.astype(int)
# duration needs to be larger than 15min
dauern = kostra.index[kostra.index >= 15]

name_place = 'Gievenbeck'
save_inp_path = os.path.join(current_path, 'pythonProject\\inp')
euler_typ = 2

# inp = euler_to_inp(inp_base,kostra, return_period=jaerlichkeiten[0], duration=dauern[0], interval=5, euler_typ=2, start_time='2024-01-01 00:00')
# print(inp[sections.TIMESERIES].to_inp_lines())



#get all euler model rain series for all return periods and durations
for j in jaerlichkeiten:
    for d in dauern:
        inp = euler_to_inp(inp_base,kostra, return_period=j, duration=d, interval=5, euler_typ=euler_typ, start_time='2024-01-01 00:00')
        inp.write_file(os.path.join(save_inp_path,f'{name_place}_e{euler_typ}_T{int(j)}D{int(d)}.inp'))












# gievenbeck = swmm_api.read_inp_file('pythonProject\\swmm_Gievenbeck.inp')





# csv_file = StringIO(""" ,TS1
# 0:05,1.7
# 0:10,2.3
# 0:15,3.4
# 0:20,8
# 0:25,1.15 """)

# series = read_csv(csv_file, index_col=0).squeeze("columns")
# series