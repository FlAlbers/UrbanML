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
from pyswmm import Nodes, Simulation, Output, NodeSeries, SystemSeries
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

def multi_node(folder_path, nodes = None, resample = '5min', threshold_multiplier = 0, min_duration = 60, accum_precip = False, storage = None):
    '''
    Extract flow data of one or multiple nodes from .out files and resample the data to a given time interval.

    Parameters:
        - folder_path: path to the folder containing the .out files
        - nodes: list of node names like in inp files
        - resample: resample time
            - example -> '5min' or '1min' ...
        - threshold_multiplier: multiplier of the max output value for calculating the threshold for minimal values at the end of each event
            - range: 0 to 1
            - example -> 0.01 for 1 % of the maximum value
                - if the max output value is 3, and the threshold_multiplier is 0.01, the threshold is 0.03. If the value is below the threshold, it is set to 0 
                - if the threshold is set to 0, no threshold is applied
        - min_duration: minimal duration of the event in minutes

    Dataoutput:
        - duration - event duration [min] - duration is negative during the start buffer where no precipitation is present
        - p - rainfall [mm/h]
        - R... - total_inflow [m³/s] - data for each node like: R0019769,  R0019768, ...

    Returns:
        - list of data for each simulation
    '''
    # if nodes is None:
    if nodes is None:
        first_file = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.inp')][0]
        with Simulation(os.path.join(folder_path, first_file)) as sim:
            nodes = [node.nodeid for node in Nodes(sim)]

    nodes = [nodes] if not isinstance(nodes, list) else nodes
    if storage is not None:
        storage = [storage] if not isinstance(storage, list) else storage

    sims_data = []
    wd = os.getcwd()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.out'):
            with Output(os.path.join(folder_path, file_name)) as out:
                # get precipitation data
                precip = SystemSeries(out).rainfall
                current_sim = pd.DataFrame({'p': precip.values()}, index=precip.keys())
                # get outfall flow data for each node
                for node in nodes:
                    outfall_flow = NodeSeries(out)[node].total_inflow
                    current_sim[node] = outfall_flow.values()
                
                if storage is not None:
                    for stor in storage:
                        stor_height = NodeSeries(out)[stor].invert_depth
                        current_sim[stor] = stor_height.values()


                # resample data
                current_sim = current_sim.resample(resample, origin = 'end').mean()
                if accum_precip:
                    current_sim['ap'] = current_sim['p'].cumsum()

                start_event = current_sim[current_sim['p'] > 0].index[0]
                current_sim['duration'] = (current_sim.index - start_event).total_seconds() / 60
                current_sim = current_sim[['duration'] + list(current_sim.columns[:-1])]
                sims_data.append((file_name, current_sim))

    # 1 % Threshold  after end of precipitation event
    if threshold_multiplier > 0:
        max_values = pd.DataFrame(columns=nodes)
        for sim in sims_data:
            max_val = sim[1][nodes].values.max(axis=0)
            max_val = max_val.reshape(1, -1)
            max_values = pd.concat([max_values,pd.DataFrame(max_val, columns=nodes)], axis=0)

        # set threshold
        thresholds = max_values.max() * threshold_multiplier

        last_p_durs = pd.DataFrame(columns=['duration'])

        # set flow values below threshold to 0 and remove data after last relevant data point
        for sim in sims_data:
            for node in nodes:
                sim[1][node][sim[1][node] < thresholds[node]] = 0
            # get last relevant time step
            last_p = pd.DataFrame([sim[1][sim[1]['p'] > 0]['duration'].iloc[-1]], columns=['duration'])
            last_p_durs = pd.concat([last_p_durs, last_p])

        # remove data after last relevant time step
        for i, (sim_id, sim_df) in enumerate(sims_data):
            first_p = pd.DataFrame([sim[1][sim[1]['p'] > 0]['duration'].iloc[0]], columns=['duration'])
            
            min_dur = first_p['duration'][0] + min_duration
            updated_df = sim_df[(sim_df['duration'] <= last_p_durs['duration'].iloc[i]) | (sim_df[nodes] > 0).any(axis=1) | (sim_df['duration'] <= min_dur)]
            sims_data[i] = (sim_id, updated_df)
    
    return sims_data

# Test Area for testing the functions
if __name__ == '__main__':

    threshold = 0.01
    folder_path = os.path.join('03_sim_data', 'inp_RR')
    nodes = ['R0019769', 'W1']
    storage = ['RR1']
    sims_data = multi_node(folder_path, nodes = nodes,resample = '5min', min_duration = 60, storage = storage)
    # sims_data = single_node(folder_path, 'R0019769',resample = '5min')
    for i in range(len(sims_data)):
        print(sims_data[i][1]['RR1'].max())
    # sims_data[0][1]['RR1'].sum()

    

    sims_data_single = single_node(folder_path, 'R0019769',resample = '5min')

    print(sims_data_single[0][1].sum())
    print(sims_data[0][1].sum())
    #ist gleich

    ##############################################################
    # Test Plot
    for sim_data in sims_data:
        plt.plot(sim_data[1].index, sim_data[1].values)

    
    plt.xlabel('Time')
    plt.ylabel('Outfall Flow')
    plt.title('Outfall Flow')
    plt.legend([sim_data[0] for sim_data in sims_data])
    plt.show()
    

