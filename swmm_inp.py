"""
Author: Flemming Albers

This script generates inp files for swmm-simulations of model rain and normal rainfall events.
The script reads a base inp file, which contains the initial configuration of the SWMM model.
It then generates multiple inp files by adding different rain series to the base inp file.

The script requires the following input parameters:
- base_inp_path: Path to the base inp file.
- event_data_path: Path to the folder containing the rain event data.
- kostra_data_path: Path to the Kostra data file.
- max_duration: Maximum duration time (in minutes) for Kostra data.
- name_place: Name of the study area.
- save_inp_path: Path to save the generated inp files.
- euler_typ: Euler type for Kostra data.
- start_time: Start time of the simulation.
- buffer_time: Buffer time before and after the rainfall event.
- TSnameKostra: Name of the Kostra time series to be included in the inp file.
- TSnameEvent: Name of the measured time series to be included in the inp file.
- TSinterval: Time interval of the time series in minutes.

The script performs the following steps:
1. Reads the Kostra data from the specified file.
2. Reads the base inp file.
3. Updates the OPTIONS section of the inp file with the specified start time, end time, and number of CPU cores.
4. Generates inp files for different combinations of return periods and durations.
    - For each combination, it adds the Kostra rain series to the inp file.
    - Updates the rain gauge for each subcatchment in the inp file.
    - Writes the modified inp file to the specified save path.
5. Generates inp files for each measured rain event in the specified event data folder.
    - For each event, it adds the measured rain series to the inp file.
    - Updates the rain gauge for each subcatchment in the inp file.
    - Writes the modified inp file to the specified save path.
"""

import os
import pandas as pd
import swmm_api
from swmm_api.input_file import section_labels as sections
from swmm_api.input_file.sections.others import RainGage, TimeseriesData
import multiprocessing

from modules.inp_helpers import euler_to_inp, event_to_inp

# get current path of working directory
current_path = os.getcwd()

###########################################################################################################
# Input section
## Input parameters for inp file generation
base_inp_path = '03_sim_data\\swmm_Gievenbeck.inp'
# path to folder with rain event data
event_data_path = '02_input_data\\events_FMO'
# path to kostra data
kostra_data_path = os.path.join(current_path, '02_input_data\\kostra_118111.csv')
# set maximum duration time [min] for Kostra data
max_duration = 72*60
# Name of the study area
name_place = 'Gievenbeck'
# Path to save the inp files
save_inp_path = os.path.join(current_path, '03_sim_data\\inp')
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

inp = inp_base
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
