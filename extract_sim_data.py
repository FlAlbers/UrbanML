#
# https://www.youtube.com/watch?v=uMF1vSbF_tk

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Nodes, Links, Output, NodeSeries, SystemSeries
import os
import pandas as pd




folder_path = '03_sim_data\\sim_test'
test_path = '03_sim_data\\sim_test\\Gievenbeck_2014-05-23 07 30 00_hN7 01.out'

sims_data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.out'):
        with Output(os.path.join(folder_path, file_name)) as out:
            outfall_flow = NodeSeries(out)['R0019769'].total_inflow
            precip = SystemSeries(out).rainfall
            
            current_sim = pd.DataFrame({'Q_out': outfall_flow.values(), 'p': precip.values()},index=outfall_flow.keys())

            # current_sim = pd.Series(outfall_flow.values(), index=outfall_flow.keys(), name='Q_out')
            # current_sim['p'] = precip.values()
            sims_data.append((file_name, current_sim))

sims_data[0]






##############################################################
# Plotting
for sim_data in sims_data:
    plt.plot(sim_data[1].index, sim_data[1].values)

  
plt.xlabel('Time')
plt.ylabel('Outfall Flow')
plt.title('Outfall Flow')
plt.legend([sim_data[0] for sim_data in sims_data])
plt.show()










# with Output(test_path) as out:
#     # outfall_flow = out.node_series.total_inflow('R0019769')
#     # outfall_flow = NodeSeries(out)['R0019769'].total_inflow
#     outfall_flow = SystemSeries(out).outfall_flows



# value at datetime.datetime(2024, 1, 1, 5, 0): 0.02081773243844509





x = outfall_flow.keys()
y = outfall_flow.values()


plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Outfall Flow')
plt.title('Outfall Flow')
plt.show()

