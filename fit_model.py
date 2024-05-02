
'''
- Einzelene Ereignisse werden nicht getrennt betrachtet, sondern die jeweiligen Sequencen die in der Zeitreihe enthalten sind
    werden zusammengeführt.
- Dimensionen in Training und prediction müssen gleich bleiben
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from keras.backend import clear_session
from modules.sequence_and_normalize import sequence_data, sequence_sample_random, sequence_list
from modules.save_load_model import save_model, load_model, save_model_container, save_model_containerOLD, load_model_container, load_model_containerOLD
from modules.extract_sim_data import multi_node, single_node
import os
from keras.models import clone_model
import time

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

'''
Data:
    R... = total_inflow [m³/s] example: R0019769
    p = rainfall [mm/h]
'''

def fit_model(model_name, save_folder, sims_data, model_init, test_size = 0.1, cv_splits = 5, 
              lag = None, delay = None, p_steps = None, in_vars = None, out_vars = None, 
              seed_train_val_test = None, seed_train_val = None, shuffle = True, loss = 'mse', epochs = 20):

    total_start_time = time.time()
    out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]
    
    train_val_data, test_data = train_test_split(sims_data, test_size=test_size, random_state=seed_train_val_test)
        
    def set_model():
        clear_session()
        # model = Model()
        model = clone_model(model_init)
        model.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae'])
        model.summary()
        return model

    ################# Training models with cross validation
    models = []
    cv_scores = pd.DataFrame(columns=['loss', 'val_loss'])
    fold_nr = 0
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed_train_val)

    if cv_splits > 1:
        
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
            # model = shuffle_weights(model)
            
            start_train = time.time()
            lstm = model.fit(x_train, y_train,epochs=epochs,batch_size=10,validation_data=(x_val, y_val),verbose=2,shuffle=shuffle)
            end_train = time.time()
            train_time = end_train - start_train

            model_copy = clone_model(model)
            model_copy.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae'])
            model_copy.set_weights(model.get_weights())
            model_container = {
                'name' : model_name,
                'model': model_copy,
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
                'history': lstm.history,
                'train_time': train_time
            }

            models.append(model_container)
            loss_last = lstm.history['loss'][-1]
            val_loss_last = lstm.history['val_loss'][-1]
            new_row = pd.DataFrame({'loss': [loss_last], 'val_loss': [val_loss_last]})
            cv_scores = pd.concat([cv_scores, new_row], ignore_index=True)

            fold_nr += 1

        for fold_id in range(len(models)):
            print(f"Fold: {fold_id}, loss = {cv_scores['loss'][fold_id]}, val_loss = {cv_scores['val_loss'][fold_id]}")

        # Select the best Model
        select_id = cv_scores['val_loss'].idxmin()

        
        # model_container = models[select_id]
        # model_container['cv_scores'] = cv_scores
        # model_container['cv_models']= models
        # model_container['select_id'] = select_id

        model_dict = {
            f'model_{i}': models[i] for i in range(len(models))
        }
        model_dict['cv_scores'] = cv_scores
        model_dict['select_id'] = select_id


#################################################################################################################
    # Resume Training with the best model
    ############### Fitting scalers for Normalization of data
    # Concatenate all data from all list objects in sims_data JUST for fitting the scalers and not for further processing
    in_concat = np.array(pd.concat([sample[1][['duration','p']] for sample in train_val_data], axis=0))
    out_concat  = np.array(pd.concat([sample[1][out_vars] for sample in train_val_data], axis=0))

    # Fitting the scalers for in and out data
    in_scaler = MinMaxScaler(feature_range=(0, 1))
    out_scaler = MinMaxScaler(feature_range=(0, 1))
    in_scaler = in_scaler.fit(in_concat)
    out_scaler = out_scaler.fit(out_concat)


    ################# Make sequences out of the data
    x_dev, y_dev = sequence_data(train_val_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                        out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

    x_test, y_test = sequence_data(test_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                        out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

    if cv_splits < 2:
        selected_model = set_model()
    else:
        selected_model = model_dict[f'model_{select_id}']['model']
        selected_model.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae'])
        selected_model.set_weights(model_dict[f'model_{select_id}']['model'].get_weights())

    start_train = time.time()
    lstm = selected_model.fit(x_dev, y_dev,epochs=60,batch_size=10,validation_data=(x_test, y_test),verbose=2,shuffle=shuffle)
    end_train = time.time()
    total_end_time = time.time()
    train_time = end_train - start_train + model_dict[f'model_{select_id}']['train_time']
    total_train_time = total_end_time - total_start_time
    model_container = {
            'name' : model_name,
            'model': selected_model,
            'in_scaler': in_scaler,
            'out_scaler': out_scaler,
            'train_data': train_val_data,
            # 'validation_data': val_data,
            'test_data': test_data,
            'lag': lag,
            'delay': delay,
            'prediction_steps': p_steps,
            'seed_train_val_test': seed_train_val_test,
            'seed_train_val': seed_train_val,
            'in_vars': in_vars,
            'out_vars': out_vars,
            'history': lstm.history,
            'train_time': train_time,
            'total_train_time': total_train_time
        }

    model_dict['selected_model'] = model_container
    return model_dict
# Load the model, the scalers and the test data
# model, in_scaler, out_scaler, train_data, val_data, test_data, data_info_dict = load_model(save_folder)
    # model_container = load_model_container(save_folder)


if __name__ == '__main__':
    model_name = 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    save_folder = os.path.join('05_models', model_name)
    folder_path_sim = os.path.join('03_sim_data', 'inp_1d_max')

    interval = 5
    lag = int(2 * 60 / interval)
    delay = -12
    p_steps = 12
    min_duration = p_steps * interval
    in_vars=['duration', 'p']
    seed_train_val_test = 8
    seed_train_val = 50
    cv_splits = 2
    sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']
    shuffle = True
    # Splitting data into train and test sets
    test_size=0.1
    epochs = 2

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

    # model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    # model.summary()

    # Train the model
    model_container = fit_model(model_name = model_name, save_folder= save_folder, sims_data= sims_data, 
                            model_init = model, test_size = test_size, cv_splits = cv_splits, lag = lag, 
                            delay = delay, p_steps = p_steps, in_vars = in_vars, out_vars = None , 
                            seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs)
    
    model_container['selected_model']['total_train_time']