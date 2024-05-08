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


def test_RR():
    model_name = 'Gievenbeck_RR_20240507'
    interval = 5
    lag = int(2 * 60 / interval)
    delay = -12
    p_steps = 12
    min_duration = p_steps * interval
    node = 'R0019769'
    in_vars_future=['duration', 'p']
    # in_vars_past = [node]
    in_vars = None
    seed_train_val_test = 8
    seed_train_val = 50
    # cv_splits = 5
    cv_splits = 5
    loss = 'mse'
    epochs = 10
    sel_epochs = 10
    units = 128
    model_folder = os.path.join('05_models', model_name)
    folder_path_sim = os.path.join('03_sim_data', 'inp_RR')
    sims_data = multi_node(folder_path_sim, node,resample = '5min', threshold_multiplier=0, min_duration=min_duration) # ['R0019769','R0019717']

    # Splitting data into train and test sets
    test_size=0.1

    model = Model()

    # Define model layers.
    input_layer = Input(shape=(lag, len(in_vars_future))) # input shape: (sequence length, number of features)
    lstm_1 = LSTM(units=units, activation='relu')(input_layer) #units = number of hidden layers
    y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_1)

    # # For Second output
    # model = Model(inputs=input_layer, outputs=[y1_output, y2_output])
    model = Model(inputs=input_layer, outputs=y1_output)


    model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, 
                                model_init = model, test_size = test_size, cv_splits = cv_splits, lag = lag, 
                                delay = delay, p_steps = p_steps, in_vars_future = in_vars_future, out_vars = None , 
                                seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, epochs=epochs, loss=loss, sel_epochs = sel_epochs)
    # Save the model container
    save_model_container(model_container, model_folder)

test_RR()