from modules.sequence_and_normalize import sequence_data, sequence_sample_random, sequence_list
from modules.extract_sim_data import multi_node
from fit_model import fit_model

import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM
import tensorflow as tf

model_name = 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
model_folder = os.path.join('05_models', model_name)
folder_path_sim = os.path.join('03_sim_data', 'inp_1d_max')

interval = 5
lag = int(2 * 60 / interval)
delay = -12
p_steps = 12
min_duration = p_steps * interval
in_vars=['duration', 'p']
seed_train_val_test = 8
seed_train_val = 50
cv_splits = 5
sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

# Splitting data into train and test sets
test_size=0.1

####### Define Model
model = Model()

# Define model layers.
input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
lstm_1 = LSTM(units=32, activation='relu')(input_layer) #units = number of hidden layers
y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_1)

# # For second output define the second dense layer and the second output
# second_dense = Dense(units=32, activation='relu')(input_layer)
# # Y2 output will be fed from the second dense
# second_flatten = Flatten()(second_dense)
# y2_output = Dense(units=p_steps, name='Q2')(second_flatten)

# Define the model with the input layer and a list of output layers
model = Model(inputs=input_layer, outputs=y1_output)

# # For Second output
# model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
model.summary()

# Train the model
fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, model_init = model, test_size= test_size, cv_splits= cv_splits, lag= lag, delay= delay, p_steps= p_steps, in_vars= in_vars, out_vars= None , seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val)
