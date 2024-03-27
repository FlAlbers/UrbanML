
#########################################################################
# Create Sequences for Time Series Forecasting

# https://medium.com/nerd-for-tech/how-to-prepare-time-series-data-for-lstm-rnn-data-windowing-8d44b63a29d5
'''
- Einzelene Ereignisse werden nicht getrennt betrachtet, sondern die jeweiligen Sequencen die in der Zeitreihe enthalten sind
    werden zusammengeführt.
- Dimensionen in Training und prediction müssen gleich bleiben
'''

import numpy as np
import pandas as pd
from modules.extract_sim_data import single_node
from sklearn.preprocessing import MinMaxScaler

folder_path = '03_sim_data\\inp'
sims_data = single_node(folder_path, 'R0019769',resample = '5min')
# test = single_node(folder_path, 'R0019769')
# sims_data[1][1]['Q_out'].values

'''
Window parameters:
l = lag
d = delay
n = next steps
k = number of sequences in sample

Data:
Q_out = total_inflow [m³/s]
p = rainfall [mm/h]
'''

# sims_data = single_node(folder_path='03_sim_data\\sim_test', node= 'R0019769')

in_data = np.array([])
out_data = np.array([])
for sample in sims_data:
    # print(sample[1])
    sample_in = np.array(sample[1][['duration','p']])
    sample_out = np.array(sample[1][['Q_out']])

    ###################################
    '''
    https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/
    https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    standard scalar or normalize needed for LSTM
    use MinMaxScaler for normalize 
    '''
    ######### windowing
    # 3h
    # l = 3h * 60 min / 5 min
    l = int(3 * 60 / 5)
    d = 3
    n = 5

    # Calculate Total number of sequences k from one sample 
    N = sample_in.shape[0] 
    k = N - (l + d + n)

    # Preapare Input and output Slice
    in_slice = np.array([range(i, i + l) for i in range(k)])
    out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])
    
    # Slice sequences from sample
    if in_data.size == 0:
        in_data = sample_in[in_slice,:]
        out_data = sample_out[out_slice,:]
    else:
        in_data = np.append(in_data, sample_in[in_slice,:], axis=0)
        out_data = np.append(out_data, sample_out[out_slice,:], axis=0)

print(out_data.shape)




####################################################
# test area
view5 = sims_data[1][1]
view1 = test[1][1]

view1[(view1.index >= '2024-01-01 07:30:00') & (view1.index <= '2024-01-01 07:40:00')]
view5[(view5.index >= '2024-01-01 07:30:00') & (view5.index <= '2024-01-01 07:40:00')]

mean_sims_data = np.mean(sims_data[1][1])
mean_test = np.mean(test[1][1])








# Example DataFrame for Store A
data_a = {'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
          'Sales': [100, 150, 130, 120, 160],
          'Visitors': [50, 60, 55, 45, 65]}
df_a = pd.DataFrame(data_a).set_index('Date')

def create_sequences(df, sequence_length):
    X, Y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:(i + sequence_length), :].values)
        Y.append(df.iloc[i + sequence_length, :].values)
    return np.array(X), np.array(Y)

# Using a sequence length of 3
sequence_length = 3
X, Y = create_sequences(df_a[['Sales', 'Visitors']], sequence_length)

print("X shape:", X.shape)  # (samples, timesteps, features)
print("Y shape:", Y.shape)  # (samples, features)