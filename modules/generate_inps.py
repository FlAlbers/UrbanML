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

def generate_inps(inp_dict):
    base_inp_path = inp_dict['base_inp_path']
    event_data_path = inp_dict['event_data_path']
    kostra_data_path = inp_dict['kostra_data_path']
    max_duration = inp_dict['max_duration']
    name_place = inp_dict['name_place']
    save_inp_path = inp_dict['save_inp_path']
    euler_typ = inp_dict['euler_typ']
    start_time = inp_dict['start_time']
    buffer_time = inp_dict['buffer_time']
    TSnameKostra = inp_dict['TSnameKostra']
    TSnameEvent = inp_dict['TSnameEvent']
    TSinterval = inp_dict['TSinterval']
    cpu_cores = inp_dict['cpu_cores']
    report_subcatchments = inp_dict['report_subcatchments']



    # # get current path of working directory
    # current_path = os.getcwd()

    # ###########################################################################################################
    # # Input section
    # ## Input parameters for inp file generation
    # base_inp_path = os.path.join('03_sim_data', 'Gievenbeck_20240325.inp')
    # # base_inp_path = '03_sim_data\\Gievenbeck_20240325.inp'
    # # path to folder with rain event data
    # event_data_path = os.path.join('02_input_data', 'events_FMO')
    # # event_data_path = '02_input_data\\events_FMO'
    # # path to kostra data
    # kostra_data_path = os.path.join(current_path, '02_input_data', 'kostra_118111.csv')
    # # set maximum duration time [min] for Kostra data
    # max_duration = 24*60 # 24 hours
    # # Name of the study area
    # name_place = 'Gievenbeck'
    # # Path to save the inp files
    # save_inp_path = os.path.join(current_path, '03_sim_data','inp_1d_max')
    # # Euler type for Kostra data (2 is standard)
    # euler_typ = 2
    # # Start time of the simulation
    # start_time = pd.to_datetime('2024-01-01 00:00')
    # # Buffer time before and after the rainfall event
    # buffer_time = pd.Timedelta('1h')
    # # Name of the Kostra time series to be included in the inp file
    # TSnameKostra = 'Kostra'
    # # Name of the measured time series to be included in the inp file
    # TSnameEvent = 'FMO'
    # # Time interval of the time series in minutes
    # TSinterval = 5
    # # amount of cpu cores in the system
    # cpu_cores = multiprocessing.cpu_count()
    # # slect if subcatchment values should be reported TRUE or FALSE
    # report_subcatchments = False

    # # End of input section
    # ############################################################################################################

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
    if not report_subcatchments:
        del inp_base[sections.REPORT]['SUBCATCHMENTS']

    ################################################################################################################
    # create inp files
    # get all euler model rain series for all return periods and durations
    if not os.path.exists(save_inp_path):
        os.mkdir(save_inp_path)

    for j in returnrate:
        for d in durations:
            inp = inp_base
            inp['TITLE'] = f'{name_place}_e{euler_typ}_T{int(j)}D{int(d)}' 
            inp = euler_to_inp(inp,kostra, return_period=j, duration=d, interval=5, euler_typ=euler_typ, start_time=start_time, TSname=TSnameKostra, buffer_time = buffer_time)
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
            inp = event_to_inp(inp, event_data, start_time=start_time, TSname=TSnameEvent, buffer_time = buffer_time)
            for subcatchment in inp['SUBCATCHMENTS']:
                inp['SUBCATCHMENTS'][subcatchment].rain_gage = TSnameEvent
            file_name = file_name.replace('.csv', '')
            file_name = file_name.replace('.', ' ')
            inp['TITLE'] = f'{name_place}_{file_name}'
            inp.write_file(os.path.join(save_inp_path,f'{name_place}_{file_name}.inp'))
    return 'All Inp files generated in ' + save_inp_path

if __name__ == '__main__':
        
    # get current path of working directory
    current_path = os.getcwd()
    # initialize dictionary for input parameters
    inp_dict = {}
    ###########################################################################################################
    # Input section
    ## Input parameters for inp file generation
    inp_dict['base_inp_path'] = os.path.join('03_sim_data', 'Gievenbeck_20240325.inp')
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
    inp_dict['save_inp_path'] = os.path.join(current_path, '03_sim_data','inp_1d_max')
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

