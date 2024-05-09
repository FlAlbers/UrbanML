from modules.sequence_and_normalize import sequence_data, sequence_sample_random, sequence_list
from modules.extract_sim_data import multi_node
from fit_model import fit_model

import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM
import tensorflow as tf
from modules.save_load_model import save_model_container, load_model_container
from datetime import date


def units_compare():
    # Model 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    # loss_functions = ['mse', 'mae', 'mape']
    model_names = ['Gievenbeck_LSTM_Single_MSE_u128' + '_' +str(date.today())]
    model_folders = []
    for m_name in model_names:
        model_folders.append(os.path.join('05_models/test_wehr', m_name))

    folder_path_sim = os.path.join('03_sim_data', 'inp_RR')

    interval = 5
    lag = int(2 * 60 / interval)
    delay = -12
    p_steps = 12
    min_duration = p_steps * interval
    in_vars=['duration', 'p', 'ap']
    out_vars = ['W1']
    storage = ['RR1']
    seed_train_val_test = 8
    seed_train_val = 50
    cv_splits = 5
    shuffle = True
    epochs = 20
    loss = 'mse'
    units = [128]
    sims_data = multi_node(folder_path_sim, ['W1'],resample = '5min', min_duration=min_duration, accum_precip=True, storage=storage) # ['R0019769','R0019717']

    # Splitting data into train and test sets
    test_size=0.1

    for model_name, model_folder, units_val in zip(model_names, model_folders, units):
    ####### Define Model
        model = Model()

        # Define model layers.
        input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
        lstm_1 = LSTM(units=units_val, activation='relu')(input_layer) #units = number of hidden layers
        y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_1)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=y1_output)

        # Train the model
        model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, model_init = model, 
                                    test_size = test_size, cv_splits = cv_splits, lag = lag, delay = delay, p_steps = p_steps, 
                                    in_vars_future = in_vars, out_vars = out_vars , seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs, loss=loss)
        # Save the model container
        save_model_container(model_container, model_folder)
units_compare()