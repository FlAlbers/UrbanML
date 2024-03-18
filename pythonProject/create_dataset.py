#
# https://www.youtube.com/watch?v=uMF1vSbF_tk

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Nodes, Links, Output
import os

folder_path = '.\\pythonProject\\sim_test'

#extract data from out file
outfall_flow = {}
node_head_outfile = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith('.out'):
        with Output(os.path.join(folder_path, file_name)) as out:
            outfall_flow[file_name] = out.node_series('R0019769', 'Flow')
            # node_head_outfile[file_name] = out.node_series('R0019466', 'Depth')



#extract precipitation data from inp file
#append data from all sims to one df
#list available nodes


x = outfall_flow.keys()
y = outfall_flow.values()


plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Outfall Flow')
plt.title('Outfall Flow')
plt.show()


