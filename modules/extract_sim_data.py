# Tutorial: SWMM output
# https://www.youtube.com/watch?v=uMF1vSbF_tk

# pyswmm api documentation
# https://pyswmm.github.io/pyswmm/reference/

#test_path = '03_sim_data\\sim_test\\Gievenbeck_2014-05-23 07 30 00_hN7 01.out'

'''
Author: Flemming Albers
Date: 11.04.2024

Description:
These functions are used to extract simulation data from .out files. The data is then resampled and stored in a pandas DataFrame.
'''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Nodes, Links, Output, NodeSeries, SystemSeries
import os
import pandas as pd


def single_node(folder_path, node = 'R0019769', resample = '1min'):
    '''
    Extract flow data of a single node from .out files and resample the data to a given time interval.

    Parameters:
        - folder_path: path to the folder containing the .out files
        - node: node name like in inp files
        - resample: resample time
            - example -> '5min' or '1min' ...

    Dataoutput:
        - duration - event duration [min] - duration is negative during the start buffer where no precipitation is present
        - p - rainfall [mm/h]
        - Q_out - total_inflow [m³/s]

    Returns:
        - list of with data for each simulation
    '''

    sims_data = []
    wd = os.getcwd()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.out'):
            with Output(os.path.join(folder_path, file_name)) as out:
                outfall_flow = NodeSeries(out)[node].total_inflow
                precip = SystemSeries(out).rainfall
                
                current_sim = pd.DataFrame({'Q_out': outfall_flow.values(), 'p': precip.values()}, index=outfall_flow.keys())
                current_sim = current_sim.resample(resample, origin = 'end').mean()
                start_event = current_sim[current_sim['p'] > 0].index[0]
                current_sim['duration'] = (current_sim.index - start_event).total_seconds() / 60
                current_sim = current_sim[['duration', 'p', 'Q_out']]
                sims_data.append((file_name, current_sim))

    return sims_data

def multi_node(folder_path, nodes = None, resample = '5min'):
    '''
    Extract flow data of a multiple nodes from .out files and resample the data to a given time interval.

    Parameters:
        - folder_path: path to the folder containing the .out files
        - nodes: nodes names like in inp files
        - resample: resample time
            - example -> '5min' or '1min' ...

    Dataoutput:
        - duration - event duration [min] - duration is negative during the start buffer where no precipitation is present
        - p - rainfall [mm/h]
        - R... - total_inflow [m³/s] - data for each node like: R0019769,  R0019768, ...

    Returns:
        - list of with data for each simulation
    '''
    sims_data = []
    wd = os.getcwd()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.out'):
            with Output(os.path.join(folder_path, file_name)) as out:
                precip = SystemSeries(out).rainfall
                current_sim = pd.DataFrame({'p': precip.values()}, index=precip.keys())
                for node in nodes:
                    outfall_flow = NodeSeries(out)[node].total_inflow
                    current_sim[node] = outfall_flow.values()
                
                current_sim = current_sim.resample(resample, origin = 'end').mean()
                start_event = current_sim[current_sim['p'] > 0].index[0]
                current_sim['duration'] = (current_sim.index - start_event).total_seconds() / 60
                current_sim = current_sim[['duration'] + list(current_sim.columns[:-1])]
                sims_data.append((file_name, current_sim))

    return sims_data

                

# Test Area for testing the functions
if __name__ == '__main__':
    folder_path = os.path.join('03_sim_data', 'sim_test')
    nodes = ['R0019769', 'R0019768']
    sims_data = multi_node(folder_path, nodes,resample = '5min')
    # sims_data = single_node(folder_path, 'R0019769',resample = '5min')
    sims_data

    ##############################################################
    # Test Plot
    for sim_data in sims_data:
        plt.plot(sim_data[1].index, sim_data[1].values)

    
    plt.xlabel('Time')
    plt.ylabel('Outfall Flow')
    plt.title('Outfall Flow')
    plt.legend([sim_data[0] for sim_data in sims_data])
    plt.show()

