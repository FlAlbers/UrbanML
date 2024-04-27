
'''
- Einzelene Ereignisse werden nicht getrennt betrachtet, sondern die jeweiligen Sequencen die in der Zeitreihe enthalten sind
    werden zusammengeführt.
- Dimensionen in Training und prediction müssen gleich bleiben
'''

import numpy as np
import pandas as pd
from modules.extract_sim_data import multi_node, single_node
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM
from keras.backend import clear_session
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from modules.sequence_and_normalize import sequence_data, sequence_sample_random, sequence_list
from modules.save_load_model import save_model, load_model, save_model_containerOLD
import os

# import modules.save_load_model
# import importlib
# importlib.reload(modules.save_load_model)

'''
Window parameters:
    To set:
        l       = lag               = number of time steps for the input sequence
        d       = delay             = number of time steps to move start of prediction window
        p_steps = prediction steps  = number of time steps to predict

    Will be calculated:
        k       = number of sequences in one sample
'''

interval = 5
lag = int(2 * 60 / interval)
delay = -12
p_steps = 12

min_duration = p_steps * interval
folder_path_sim = os.path.join('03_sim_data', 'inp_1d_max')
sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

model_name = 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
# model_name = 'Gievenbeck_LSTM_Single_Thresh_1h_P_20240408'
model_folder = os.path.join('05_models', model_name)

seed_train_val_test = 8
seed_train_val = 50
# Splitting data into train and test sets
train_val_data, test_data = train_test_split(sims_data, test_size=0.1, random_state=seed_train_val_test)


# Set column names for input and output variables
in_vars=['duration', 'p']
out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]
# out_vars=['Q_out']


'''
Data:
    R... = total_inflow [m³/s] example: R0019769
    p = rainfall [mm/h]
'''

def set_model():
    clear_session()
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

    return model

################# Training models with cross validation
cv_splits = 5
models = []
cv_scores = pd.DataFrame(columns=['loss', 'val_loss'])
fold_nr = 0
cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed_train_val)

for train, val in cv.split(train_val_data):
    train_data = [train_val_data[i] for i in train]
    val_data = [train_val_data[i] for i in val]
    
    ############### Fitting scalers for Normalization of data
    # Concatenate all data from all list objects in sims_data JUST for fitting the scalers and not for further processing
    in_concat = np.array(pd.concat([sample[1][['duration','p']] for sample in train_data], axis=0))
    out_concat  = np.array(pd.concat([sample[1][out_vars] for sample in train_data], axis=0))

    # Fitting the scalers for in and out data
    in_scaler = MinMaxScaler(feature_range=(0, 1))
    out_scaler = MinMaxScaler(feature_range=(0, 1))
    in_scaler = in_scaler.fit(in_concat)
    out_scaler = out_scaler.fit(out_concat)


    ################# Make sequences out of the data
    x_train, y_train = sequence_data(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                        out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    print(x_train.shape)
    print(y_train[0].shape)
    print(y_train[1].shape)

    x_val, y_val = sequence_data(val_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                    out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)


    # Train the model
    model = set_model()

    lstm = model.fit(x_train, y_train,epochs=20,batch_size=10,validation_data=(x_val, y_val),verbose=2,shuffle=True)

    model_container = {
        'name' : model_name,
        'model': model,
        'in_scaler': in_scaler,
        'out_scaler': out_scaler,
        'train_data': train_data,
        'validation_data': val_data,
        'test_data': test_data,
        'lag': lag,
        'delay': delay,
        'prediction_steps': p_steps,
        'seed_train_val_test': seed_train_val_test,
        'seed_train_val': seed_train_val,
        'in_vars': in_vars,
        'out_vars': out_vars,
        'history': lstm.history
    }

    models.append(model_container)
    loss = lstm.history['loss'][-1]
    val_loss = lstm.history['val_loss'][-1]
    new_row = pd.DataFrame({'loss': [loss], 'val_loss': [val_loss]})
    cv_scores = pd.concat([cv_scores, new_row], ignore_index=True)

    fold_nr += 1

for fold_id in range(len(models)):
    print(f"Fold: {fold_id}, loss = {cv_scores['loss'][fold_id]}, val_loss = {cv_scores['val_loss'][fold_id]}")

# Select the best Model
select_id = cv_scores['val_loss'].idxmin()

model_container = models[select_id]
model_container['cv_scores'] = cv_scores

# Plot the learning curve
# pyplot.plot(model_container['history']['loss'], '--', label='Training')
# pyplot.plot(model_container['history']['val_loss'], label='Validierung')
# pyplot.xlabel('Trainingsepoche')
# pyplot.ylabel('Mittlerer quadratischer Fehler [-]')
# pyplot.legend()
# pyplot.show()
###############################################################
# Saving and loading the model
save_model_containerOLD(model_container, save_folder=model_folder)

# Save the pyplot figure to the model_folder
pyplot.plot(model_container['history']['loss'], '--', label='Training')
pyplot.plot(model_container['history']['val_loss'], label='Validierung')
pyplot.xlabel('Trainingsepoche')
pyplot.ylabel('Mittlerer quadratischer Fehler [-]')
pyplot.legend()
figure_path = os.path.join(model_folder, 'learning_curve.png')
pyplot.savefig(figure_path)

# Load the model, the scalers and the test data
# model, in_scaler, out_scaler, train_data, val_data, test_data, data_info_dict = load_model(model_folder)





