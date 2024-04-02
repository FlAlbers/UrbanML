
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
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf

folder_path = '03_sim_data\\inp'
sims_data = single_node(folder_path, 'R0019769',resample = '5min')
# test = single_node(folder_path, 'R0019769')
# sims_data[1][1]['Q_out'].values


random_seed = 42
# Splitting data into train and test sets
train_val_data, test_data = train_test_split(sims_data, test_size=0.1, random_state=random_seed)

# Splitting train data again into train and validation sets
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=random_seed)

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

############ Fitting scalers for Normalization of data
# Concatenate all data from all list objects in sims_data JUST for fitting the scalers and not for further processing
in_concat = np.array(pd.concat([sample[1][['duration','p']] for sample in sims_data], axis=0))
out_concat  = np.array(pd.concat([sample[1][['Q_out']] for sample in sims_data], axis=0))

# Fitting the scalers for in and out data
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

def sequence_data(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    for sample in sims_data:
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        N = in_sample.shape[0]
        k = N - (lag + delay + prediction_steps)

        in_slice = np.array([range(i, i + l) for i in range(k)])
        out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        if in_data.size == 0:
            in_data = in_sample[in_slice, :]
            out_data = out_sample[out_slice, :]
        else:
            in_data = np.append(in_data, in_sample[in_slice, :], axis=0)
            out_data = np.append(out_data, out_sample[out_slice, :], axis=0)
    return in_data, out_data

def sequence_sample_random(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, random_seed=42):
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    rnd_indices = np.random.choice(len(sims_data))
    sample = sims_data[rnd_indices]
    in_sample = np.array(sample[1][in_vars])
    out_sample = np.array(sample[1][out_vars])
    in_sample = in_scaler.transform(in_sample)
    out_sample = out_scaler.transform(out_sample)

    N = in_sample.shape[0]
    k = N - (lag + delay + prediction_steps)

    in_slice = np.array([range(i, i + l) for i in range(k)])
    out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

    in_data = in_sample[in_slice, :]
    out_data = out_sample[out_slice, :]
    return in_data, out_data




#########################################################################
# Use Sequence function to create x and y data for train and test
lag = int(3 * 60 / 5)
delay = 0
p_steps = 6

in_vars=['duration', 'p']
out_vars=['Q_out']




x_train, y_train = sequence_data(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                    out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(x_train.shape)
print(y_train.shape)

'''
Include crossvalidation here to split the training data into training and validation data for crossvalidation
https://scikit-learn.org/stable/modules/cross_validation.html

Maybe block chaining cross validation
https://www.linkedin.com/pulse/improving-lstm-performance-using-time-series-cross-validation-mu/


'''

x_val, y_val = sequence_data(val_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(x_val.shape)
print(y_val.shape)

# Design network
model = Sequential()
model.add(LSTM(10, input_shape=(lag, len(in_vars))))
model.add(Dense(p_steps)) # dense is the number of output neurons or the number of predictions. This could also be achieved with return_sequence =ture and TimeDistributed option
model.compile(loss='mae', optimizer='adam')

# Fit network
lstm = model.fit(x_train, y_train,
epochs=60,
batch_size=10,
validation_data=(x_val, y_val),
verbose=2,
shuffle=False)

pyplot.plot(lstm.history['loss'], '--', label='train loss')
pyplot.plot(lstm.history['val_loss'], label='test loss')
pyplot.legend()
pyplot.show()

###############################################################
# Saving and loading the model
# serialize model to JSON
model_json = model.to_json()
with open("Gievenbeck_SingleNode_LSTM_20240328.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Gievenbeck_SingleNode_LSTM_20240328.weights.h5")
print("Saved model to disk")

# load json and create model
json_file = open('Gievenbeck_SingleNode_LSTM_20240328.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Gievenbeck_SingleNode_LSTM_20240328.weights.h5")
print("Loaded model from disk")

###############################################################
# Test the model

x_test, y_test = sequence_sample_random(test_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps, random_seed=random_seed)
print(x_test.shape)
print(y_test.shape)

y_test_x = y_test.reshape(y_test.shape[0], -1)

Predict = model.predict(x_test)
Predict_revert = out_scaler.inverse_transform(Predict)
y_revert = out_scaler.inverse_transform(y_test_x)

n = int(len(x_test) / 3)
Predict_revert[n]
y_revert[n]
Predict[n]
y_test[n]
# Plotting the predicted and actual values
plt.plot(Predict_revert[n], label='Predicted')
plt.plot(y_revert[n], label='Actual')
plt.ylim(bottom=0)  # Set y-axis to start from zero
plt.legend()
plt.show()

####################################################
# test area
# view5 = sims_data[1][1]
# view1 = test[1][1]

# view1[(view1.index >= '2024-01-01 07:30:00') & (view1.index <= '2024-01-01 07:40:00')]
# view5[(view5.index >= '2024-01-01 07:30:00') & (view5.index <= '2024-01-01 07:40:00')]

# mean_sims_data = np.mean(sims_data[1][1])
# mean_test = np.mean(test[1][1])








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