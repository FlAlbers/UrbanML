
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
from sklearn.model_selection import train_test_split

folder_path = '03_sim_data\\inp'
sims_data = single_node(folder_path, 'R0019769',resample = '5min')
# test = single_node(folder_path, 'R0019769')
# sims_data[1][1]['Q_out'].values

# Splitting data into train and test sets
train_data, test_data = train_test_split(sims_data, test_size=0.2, random_state=42)

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

############ Fitting scaler for Normalization of data
# Concatenate all data from all list objects in sims_data JUST for fitting the scalers and not for further processing
in_concat = np.array(pd.concat([sample[1][['duration','p']] for sample in sims_data], axis=0))
out_concat  = np.array(pd.concat([sample[1][['Q_out']] for sample in sims_data], axis=0))

in_scaler = MinMaxScaler(feature_range=(0, 1))
out_scaler = MinMaxScaler(feature_range=(0, 1))

in_scaler = in_scaler.fit(in_concat)
out_scaler = out_scaler.fit(out_concat)

### Scaler check if before and after is the same
# out_norm = out_scaler.transform(out_concat)
# out_back = out_scaler.inverse_transform(out_norm)
# sum(out_concat[:,0])
# sum(out_norm[:,0])
# sum(out_back[:,0])


def sequence_data(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None):
    in_data = np.array([])
    out_data = np.array([])

    for sample in sims_data:
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        l = int(3 * 60 / 5)
        d = 3
        n = 5

        N = in_sample.shape[0]
        k = N - (l + d + n)

        in_slice = np.array([range(i, i + l) for i in range(k)])
        out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        if in_data.size == 0:
            in_data = in_sample[in_slice, :]
            out_data = out_sample[out_slice, :]
        else:
            in_data = np.append(in_data, in_sample[in_slice, :], axis=0)
            out_data = np.append(out_data, out_sample[out_slice, :], axis=0)

    return in_data, out_data

in_train, out_train = sequence_data(train_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=in_scaler, out_scaler=out_scaler)
print(out_train.shape)
print(in_train.shape)


in_test, out_test = sequence_data(test_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=in_scaler, out_scaler=out_scaler)
print(out_test.shape)
print(in_test.shape)


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