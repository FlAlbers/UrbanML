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
import multiprocessing
from modules.generate_inps import generate_inps
from modules.swmm_ex import swmm_ex_multiprocessing as ex

# get current path of working directory
current_path = os.getcwd()
# initialize dictionary for input parameters
inp_dict = {}
###########################################################################################################
# Input section
## Input parameters for inp file generation
inp_dict['base_inp_path'] = os.path.join('03_sim_data', 'Gievenbeck_RR_20240507.inp')
# base_inp_path = '03_sim_data\\Gievenbeck_20240325.inp'
# path to folder with rain event data
inp_dict['event_data_path'] = os.path.join('02_input_data', 'events_FMO')
# event_data_path = '02_input_data\\events_FMO'
# path to kostra data
inp_dict['kostra_data_path'] = os.path.join(current_path, '02_input_data', 'kostra_118111.csv')
# set maximum duration time [min] for Kostra data
inp_dict['max_duration'] = 24*60 # 24 hours
# Name of the study area
inp_dict['name_place'] = 'Gievenbeck'
# Path to save the inp files
inp_dict['save_inp_path'] = os.path.join(current_path, '03_sim_data','inp_RR')
# Euler type for Kostra data (2 is standard)
inp_dict['euler_typ'] = 2
# Start time of the simulation
inp_dict['start_time'] = pd.to_datetime('2024-01-01 00:00')
# Buffer time before and after the rainfall event
inp_dict['buffer_time'] = pd.Timedelta('1h')
# Name of the Kostra time series to be included in the inp file
inp_dict['TSnameKostra'] = 'Kostra'
# Name of the measured time series to be included in the inp file
inp_dict['TSnameEvent'] = 'FMO'
# Time interval of the time series in minutes
inp_dict['TSinterval'] = 5
# amount of cpu cores in the system
inp_dict['cpu_cores'] = multiprocessing.cpu_count()
# slect if subcatchment values should be reported TRUE or FALSE
inp_dict['report_subcatchments'] = False

generate_inps(inp_dict)


sim_path = os.path.join('03_sim_data', 'inp_RR')

ex.swmm_mp(sim_path)