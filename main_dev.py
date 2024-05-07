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

###############################################################################
# Model 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'

def loss_functions_compare():
    loss_functions = ['mse', 'mae', 'mape']
    model_names = ('Gievenbeck_LSTM_Single_MSE'+str(date.today()), 'Gievenbeck_LSTM_Single_MAE' +str(date.today()), 'Gievenbeck_LSTM_Single_MAPE' +str(date.today()))
    model_folders = []
    for m_name in model_names:
        model_folders.append(os.path.join('05_models/loss_functions_compare', m_name))

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
    shuffle = True
    epochs = 20
    sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

    # Splitting data into train and test sets
    test_size=0.1

    for model_name, model_folder, loss in zip(model_names, model_folders, loss_functions):
        ####### Define Model
        model = Model()

        # Define model layers.
        input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
        lstm_1 = LSTM(units=32, activation='relu')(input_layer) #units = number of hidden layers
        y1_output = Dense(units=p_steps, activation='relu', name='Out')(lstm_1)
        model = Model(inputs=input_layer, outputs=y1_output)

        # # For Second output
        # model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

        # model.compile(loss=loss, optimizer='adam', metrics=['mse', 'mae', 'mape'])
        # model.summary()

        # Train the model
        model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, 
                                    model_init = model, test_size = test_size, cv_splits = cv_splits, lag = lag, 
                                    delay = delay, p_steps = p_steps, in_vars_future = in_vars, out_vars = None , 
                                    seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, loss=loss, epochs=epochs)
        # Save the model container
        save_model_container(model_container, model_folder)
loss_functions_compare()

####################################################################################################


######################################################################################################

def shuffle_compare():
    # Model 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    # loss_functions = ['mse', 'mae', 'mape']
    model_names = ('Gievenbeck_LSTM_Single_MSE_Shuffle_'+str(date.today()), 'Gievenbeck_LSTM_Single_MSE_No_Shuffle_' +str(date.today()))
    model_folders = []
    for m_name in model_names:
        model_folders.append(os.path.join('05_models/shuffle_compare', m_name))

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
    shuffle_vals = [True, False]
    epochs = 20
    loss = 'mse'
    sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

    # Splitting data into train and test sets
    test_size=0.1

    for model_name, model_folder, shuffle in zip(model_names, model_folders, shuffle_vals):
    ####### Define Model
        model = Model()

        # Define model layers.
        input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
        lstm_1 = LSTM(units=32, activation='relu')(input_layer) #units = number of hidden layers
        y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_1)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=y1_output)

        # Train the model
        model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, model_init = model, 
                                    test_size = test_size, cv_splits = cv_splits, lag = lag, delay = delay, p_steps = p_steps, 
                                    in_vars_future = in_vars, out_vars = None , seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs, loss=loss)
        # Save the model container
        save_model_container(model_container, model_folder)
shuffle_compare()

######################################################################################################

def units_compare():
    # Model 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    # loss_functions = ['mse', 'mae', 'mape']
    model_names = ['Gievenbeck_LSTM_Single_MSE_u32' + '_' +str(date.today()), 'Gievenbeck_LSTM_Single_MSE_u64' + '_' +str(date.today()),'Gievenbeck_LSTM_Single_MSE_u128' + '_' +str(date.today()),'Gievenbeck_LSTM_Single_MSE_u256' + '_' +str(date.today()),'Gievenbeck_LSTM_Single_MSE_u512' + '_' +str(date.today())]
    model_folders = []
    for m_name in model_names:
        model_folders.append(os.path.join('05_models/units_compare', m_name))

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
    shuffle = True
    epochs = 20
    loss = 'mse'
    units = [32, 64, 128, 256, 512]
    sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

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
                                    in_vars_future = in_vars, out_vars = None , seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs, loss=loss)
        # Save the model container
        save_model_container(model_container, model_folder)
units_compare()

######################################################################################################

def deep_compare():
    # Model 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    # loss_functions = ['mse', 'mae', 'mape']
    model_names = ['Gievenbeck_LSTM_Double_MSE_u64' + '_' +str(date.today()), 'Gievenbeck_LSTM_Triple_MSE_u32' + '_' +str(date.today())]
    model_folders = []
    for m_name in model_names:
        model_folders.append(os.path.join('05_models/deep_compare', m_name))

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
    shuffle = True
    epochs = 20
    loss = 'mse'
    units = [64,32]
    n_layers = [2, 3]
    sims_data = multi_node(folder_path_sim, 'R0019769',resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

    # Splitting data into train and test sets
    test_size=0.1

    for model_name, model_folder, n_layer, units in zip(model_names, model_folders, n_layers, units):
    ####### Define Model
        model = Model()

        # Define model layers.
        input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
        if n_layer == 2:
            lstm_1 = LSTM(units=units, activation='relu', return_sequences=True)(input_layer)
            lstm_2 = LSTM(units=units, activation='relu')(lstm_1) #units = number of hidden layers
            y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_2)
        elif n_layer == 3:
            lstm_1 = LSTM(units=units, activation='relu', return_sequences=True)(input_layer)
            lstm_2 = LSTM(units=units, activation='relu', return_sequences=True)(lstm_1)
            lstm_3 = LSTM(units=units, activation='relu')(lstm_2)
            y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_3)

        # Define the model with the input layer and a list of output layers
        model = Model(inputs=input_layer, outputs=y1_output)

        # Train the model
        model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, model_init = model, 
                                    test_size = test_size, cv_splits = cv_splits, lag = lag, delay = delay, p_steps = p_steps, 
                                    in_vars_future = in_vars, out_vars = None , seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs, loss=loss)
        # Save the model container
        save_model_container(model_container, model_folder)
deep_compare()



######################################################################################################
# Test
# Train the model
model_name = 'Gievenbeck_Past_double_u128'
interval = 5
lag = int(2 * 60 / interval)
delay = -12
p_steps = 12
min_duration = p_steps * interval
node = 'R0019769'
in_vars_future=['duration', 'p']
in_vars_past = [node]
in_vars = in_vars_past + in_vars_future
seed_train_val_test = 8
seed_train_val = 50
# cv_splits = 5
cv_splits = 5
loss = 'mse'
epochs = 20
sel_epochs = 60
units = 128
model_folder = os.path.join('05_models', model_name)
folder_path_sim = os.path.join('03_sim_data', 'inp_1d_max')
sims_data = multi_node(folder_path_sim, node,resample = '5min', threshold_multiplier=0.01, min_duration=min_duration) # ['R0019769','R0019717']

# Splitting data into train and test sets
test_size=0.1

model = Model()

# Define model layers.
input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
lstm_1 = LSTM(units=units, activation='relu', return_sequences=True)(input_layer)
lstm_2 = LSTM(units=units, activation='relu')(lstm_1) #units = number of hidden layers
y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_2)

# # For Second output
# model = Model(inputs=input_layer, outputs=[y1_output, y2_output])
model = Model(inputs=input_layer, outputs=y1_output)


model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, 
                            model_init = model, test_size = test_size, cv_splits = cv_splits, lag = lag, 
                            delay = delay, p_steps = p_steps, in_vars_future = in_vars_future, out_vars = None , 
                            seed_train_val_test = seed_train_val_test, seed_train_val = seed_train_val, epochs=epochs, loss=loss, sel_epochs = sel_epochs, in_vars_past = in_vars_past)
# Save the model container
save_model_container(model_container, model_folder)
# print(len(model_container['selected_model']['history']['mse']))