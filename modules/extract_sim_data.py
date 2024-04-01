# Tutorial: SWMM output
# https://www.youtube.com/watch?v=uMF1vSbF_tk

# pyswmm api documentation
# https://pyswmm.github.io/pyswmm/reference/

#test_path = '03_sim_data\\sim_test\\Gievenbeck_2014-05-23 07 30 00_hN7 01.out'


'''
Author: Flemming Albers

Extract simulation data from .out files

Parameters:
    - folder_path: path to the folder containing the .out files
    - node: node name
    - resample: resample time example -> '5min'
        -> resampling calculates looking foreward in time
Dataoutput:
 - Q_out - total_inflow [m³/s]
 - p - rainfall [mm/h]

Edited Dataoutput:
 - 5 min Mean
 - New col Duration [min] for elapsed time of event

'''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Nodes, Links, Output, NodeSeries, SystemSeries
import os
import pandas as pd



def single_node(folder_path, node = 'R0019769', resample = '1min'):
    sims_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.out'):
            with Output(os.path.join(folder_path, file_name)) as out:
                outfall_flow = NodeSeries(out)[node].total_inflow
                precip = SystemSeries(out).rainfall
                
                current_sim = pd.DataFrame({'Q_out': outfall_flow.values(), 'p': precip.values()}, index=outfall_flow.keys())
                current_sim = current_sim.resample(resample, origin = 'end').mean()
                current_sim['duration'] = (current_sim.index - current_sim.index[0]).total_seconds() / 60
                current_sim = current_sim[['duration', 'p', 'Q_out']]
                sims_data.append((file_name, current_sim))

    return sims_data


if __name__ == '__main__':
    folder_path = '03_sim_data\\sim_test'
    sims_data = single_node(folder_path, 'R0019769',resample = '5min')
    test = single_node(folder_path, 'R0019769')
 

    ##############################################################
    # Test Plot
    for sim_data in sims_data:
        plt.plot(sim_data[1].index, sim_data[1].values)

    
    plt.xlabel('Time')
    plt.ylabel('Outfall Flow')
    plt.title('Outfall Flow')
    plt.legend([sim_data[0] for sim_data in sims_data])
    plt.show()

