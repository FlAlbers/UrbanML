#
# https://www.youtube.com/watch?v=uMF1vSbF_tk

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyswmm import Nodes, Links, Output




folder_path = '03_sim_data\\sim_test'
test_path = '03_sim_data\\quick_sim\\Gievenbeck_2015-01-15 04 25 00_hN2 34.out'

#extract data from out file
outfall_flow = {}
node_head_outfile = {}



with Output(test_path) as out:
    outfall_flow = out.node_series('R0019769', 'Flow')

sum_outfall = sum(outfall_flow.values())

sum_outfall_mm = sum_outfall / 400000 * 1000
# node_series_index = out.node_series()
# print(node_series_index)

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




# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.out'):
#         with Output(os.path.join(folder_path, file_name)) as out:
#             outfall_flow[file_name] = out.node_series('R0019769', 'Flow')
#             # node_head_outfile[file_name] = out.node_series('R0019466', 'Depth')
