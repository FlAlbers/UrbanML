
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
from extract_sim_data import multi_node
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from modules.sequence_and_normalize import sequence_data, sequence_for_sequential, sequence_sample_random, sequence_list
from modules.save_load_model import save_model, load_model
import os


folder_path_sim = os.path.join('03_sim_data', 'inp_1d_max')
sims_data = multi_node(folder_path_sim, ['R0019769'],resample = '5min') # ['R0019769','R0019717']

model_name = 'Gievenbeck_DoubleNodeTest_LSTM_20240408'
model_folder = os.path.join('05_models', model_name)

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

in_vars=['duration', 'p']
out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]
# out_vars=['Q_out']

############ Fitting scalers for Normalization of data
# Concatenate all data from all list objects in sims_data JUST for fitting the scalers and not for further processing
in_concat = np.array(pd.concat([sample[1][['duration','p']] for sample in train_val_data], axis=0))
out_concat  = np.array(pd.concat([sample[1][out_vars] for sample in train_val_data], axis=0))

# Fitting the scalers for in and out data
in_scaler = MinMaxScaler(feature_range=(0, 1))
out_scaler = MinMaxScaler(feature_range=(0, 1))
in_scaler = in_scaler.fit(in_concat)
out_scaler = out_scaler.fit(out_concat)

#########################################################################
# Use Sequence function to create x and y data for train and test
lag = int(2 * 60 / 5)
delay = 0
p_steps = 6

x_train, y_train = sequence_data(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                    out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(x_train.shape)
print(y_train[0].shape)
print(y_train[1].shape)

'''
Include crossvalidation here to split the training data into training and validation data for crossvalidation
https://scikit-learn.org/stable/modules/cross_validation.html

Maybe block chaining cross validation
https://www.linkedin.com/pulse/improving-lstm-performance-using-time-series-cross-validation-mu/

'''

x_val, y_val = sequence_data(val_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(x_val.shape)
print(y_val[0].shape)
print(y_val[1].shape)

'''
1. When sequencing all output sequences need to be in one array containing all output sequences (like a list)
2. Make multiple dense output layers for each output sequence respectively
3. assign input and output layers to model
'''

########### Stacked LSTM
# https://colab.research.google.com/drive/1sZqFWkWTmv-htvL7OwiFMHX022Gy0Syf

# Define model layers.
input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
first_dense = Dense(units=32)(input_layer) #units = number of hidden layers
# Y1 output will be fed from the first dense
first_flatten = Flatten()(first_dense)
y1_output = Dense(units=p_steps, name='Q1')(first_flatten)

# # For second output define the second dense layer and the second output
# second_dense = Dense(units=32, activation='relu')(input_layer)
# # Y2 output will be fed from the second dense
# second_flatten = Flatten()(second_dense)
# y2_output = Dense(units=p_steps, name='Q2')(second_flatten)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=[y1_output])

# # For Second output
# model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
model.summary()


# Train the model
lstm = model.fit(x_train, y_train,epochs=20,batch_size=10,validation_data=(x_val, y_val),verbose=2,shuffle=False)

x_test, y_test = sequence_for_sequential(test_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(x_test.shape)
# print(y_test[1].shape)

# scores = model.evaluate(x_test, y_test , verbose=1)
# print('MSE = ', round(scores,4))
# print('MAE = ', round(scores[2],4))
# print('MAPE = ', round(scores[3],2) , '%')


# cvscores.append(scores * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

pyplot.plot(lstm.history['loss'], '--', label='train loss')
pyplot.plot(lstm.history['val_loss'], label='validation loss')
pyplot.legend()
pyplot.show()

###############################################################

# Saving the model, the scalers and the test data
save_model(model, model_folder, in_scaler, out_scaler, train_data, val_data, test_data)

# Load the model, the scalers and the test data
model, in_scaler, out_scaler, train_data, val_data, test_data = load_model(model_folder)


################################################################
# Test the model

# sequence data to list structure
seq_test = sequence_list(test_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
print(seq_test[0])

x_test, y_test = sequence_data(test_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                  out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps, random_seed=random_seed)
print(x_test.shape)
print(y_test.shape)

# y_test_x = y_test.reshape(y_test.shape[0], -1)

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

