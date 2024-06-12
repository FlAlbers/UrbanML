"""
Author: Flemming Albers

This script is the main script and is split into 3 parts: 
1. Generating inp files and running SWMM simulations.
2. Modelsetups and training of the LSTM models.
3. Evaluation of the models and comparison of the results.


1. Generating inp files and running SWMM simulations
This part generates inp files for swmm-simulations of model rain and normal rainfall events.
The script reads a base inp file, which contains the initial configuration of the SWMM model.
It then generates multiple inp files by adding different rain series to the base inp file.

The script requires the following input parameters:
- base_inp_path: Path to the base inp file.
- event_data_path: Path to the folder containing the rain event data.
- kostra_data_path: Path to the Kostra data file.
- max_duration: Maximum duration time (in minutes) for Kostra data.
- name_place: Name of the study area.
- save_inp_path: Path to save the generated inp files.
- euler_typ: Euler type for Kostra data.
- start_time: Start time of the simulation.
- buffer_time: Buffer time before and after the rainfall event.
- TSnameKostra: Name of the Kostra time series to be included in the inp file.
- TSnameEvent: Name of the measured time series to be included in the inp file.
- TSinterval: Time interval of the time series in minutes.

The script performs the following steps:
1. Reads the Kostra data from the specified file.
2. Reads the base inp file.
3. Updates the OPTIONS section of the inp file with the specified start time, end time, and number of CPU cores.
4. Generates inp files for different combinations of return periods and durations.
    - For each combination, it adds the Kostra rain series to the inp file.
    - Updates the rain gauge for each subcatchment in the inp file.
    - Writes the modified inp file to the specified save path.
5. Generates inp files for each measured rain event in the specified event data folder.
    - For each event, it adds the measured rain series to the inp file.
    - Updates the rain gauge for each subcatchment in the inp file.
    - Writes the modified inp file to the specified save path.
"""
import os
import pandas as pd
import multiprocessing
from modules.generate_inps import generate_inps
from modules.swmm_ex.swmm_ex import swmm_ex_batch as ex

# get current path of working directory
current_path = os.getcwd()
# initialize dictionary for input parameters
inp_dict = {}
###########################################################################################################
# Input section
## Input parameters for inp file generation
# path to where the the inp-files should be saved
inp_dict['save_inp_path'] = os.path.join(current_path, '03_sim_data','inp_test')
# path to base inp file that is used as template
inp_dict['base_inp_path'] = os.path.join('03_sim_data', 'Gievenbeck_20240325.inp')
# base_inp_path = '03_sim_data\\Gievenbeck_20240325.inp'
# path to folder with rain event data
inp_dict['event_data_path'] = os.path.join('02_input_data', 'events_FMO')
# path to kostra data
inp_dict['kostra_data_path'] = os.path.join(current_path, '02_input_data', 'kostra_118111.csv')
# set maximum duration time [min] for Kostra data
inp_dict['max_duration'] = 24*60 # 24 hours
# Name of the study area
inp_dict['name_place'] = 'Gievenbeck'
# Euler type for Kostra data (2 is standard)
inp_dict['euler_typ'] = 2
# Start time of the simulation
inp_dict['start_time'] = pd.to_datetime('2024-01-01 00:00')
# Buffer time before and after the rainfall event
inp_dict['buffer_time'] = pd.Timedelta('2h')
# Name of the Kostra time series to be included in the inp file
inp_dict['TSnameKostra'] = 'Kostra'
# Name of the measured time series to be included in the inp file
inp_dict['TSnameEvent'] = 'FMO'
# Time interval of the time series in minutes
inp_dict['TSinterval'] = 5
# amount of cpu cores in the system
inp_dict['cpu_cores'] = multiprocessing.cpu_count()
# slect if subcatchment values should be reported TRUE or FALSE
inp_dict['report_subcatchments'] = False

# generate inp files based on the input parameters
generate_inps(inp_dict)
# run the simulations
ex(inp_dict['save_inp_path'])


"""
2. Modelsetups and training of the LSTM models

This code trains an LSTM model for time series prediction.
The parameters are as follows:
- model_name: Name of the model.
- model_folder: Folder path to save the trained model.
- folder_path_sim: Folder path containing the simulation data.
- node: Node ID for which the prediction is performed.
- in_vars_future: List of input variables for future time steps.
- accum_precip: Boolean indicating whether to include accumulated precipitation as an input variable.
- Q_measured: Boolean indicating whether to include measured discharge as an input variable.
- interval: Time interval of the simulation data in minutes.
- resample: Time interval for resampling the simulation data.
- threshold_multiplier: Multiplier for the threshold to filter out low precipitation events.
- lag: Number of time steps for the input of the LSTM model.
- overlap: Number of time steps to overlap between input and output sequences.
- p_steps: Number of time steps to predict in the future.
- test_size: Proportion of the dataset to use as the test set.
- seed_train_val_test: Seed for splitting the dataset into train, validation, and test sets.
- seed_train_val: Seed for splitting the train set into train and validation sets.
- cv_splits: Number of cross-validation splits.
- shuffle: Boolean indicating whether to shuffle the dataset before splitting.
- epochs: Number of training epochs in the crossvalidation.
- loss: Loss function for the model.
- units: Number of units in the LSTM layers.

After executing the code, the trained model is saved in the specified model folder.

"""

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


# Essential parameters
model_name = 'Gievenbeck_LSTM_Test' + '_' +str(date.today())
model_folder = os.path.join('05_models','train_test', model_name)
node = 'R0019769'

def train_LSTM(model_name, model_folder, node):

    folder_path_sim = inp_dict['save_inp_path']

    # Data preparation parameters
    in_vars_future=['duration', 'p']
    accum_precip = True
    Q_measured = True
    interval = 5
    resample = '5min'
    threshold_multiplier = 0.01

    # Sequence parameters
    lag = int(2 * 60 / interval)
    overlap = 12
    p_steps = 12

    # Set size of the test set. Recommended: 0.1 or 0.2
    test_size=0.1
    
    # ML parameters
    cv_splits = 5
    shuffle = True
    epochs = 20
    loss = 'mse'
    units = 128
    seed_train_val_test = 8
    seed_train_val = 50
    ####### End of input section

    # Load the data from the simulations
    min_duration = p_steps * interval
    sims_data = multi_node(folder_path_sim, node,resample = resample, threshold_multiplier=threshold_multiplier, min_duration=min_duration, accum_precip=True)
    
    # set in and out variables
    in_vars_future.append('ap') if accum_precip else None
    in_vars_past = [node] if Q_measured else None
    in_vars = in_vars_past + in_vars_future if in_vars_past is not None else in_vars_future

    ####### Define Model
    model = Model()

    # Define model layers.
    input_layer = Input(shape=(lag, len(in_vars))) # input shape: (sequence length, number of features)
    lstm_1 = LSTM(units=units, activation='relu', return_sequences=True)(input_layer)
    lstm_2 = LSTM(units=units, activation='relu', return_sequences=True)(lstm_1)
    lstm_3 = LSTM(units=units, activation='relu')(lstm_2)
    y1_output = Dense(units=p_steps, activation='relu', name='Q1')(lstm_3)

    # Define the model with the input layer and a list of output layers
    model = Model(inputs=input_layer, outputs=y1_output)

    # Train the model
    delay = overlap *  -1
    model_container = fit_model(model_name = model_name, save_folder= model_folder, sims_data= sims_data, model_init = model, 
                                test_size = test_size, cv_splits = cv_splits, lag = lag, delay = delay, p_steps = p_steps, 
                                in_vars_future = in_vars_future, out_vars = [node] , seed_train_val_test = seed_train_val_test,
                                seed_train_val = seed_train_val, shuffle=shuffle, epochs=epochs, loss=loss, in_vars_past=in_vars_past)
    # Save the model container
    save_model_container(model_container, model_folder)
train_LSTM()


"""
3. Evaluation of the model
Evaluates one or multiple models and exports the results to html.

    Parameters:
    - model_names (list): A list of model names to evaluate.
    - model_alias (list): A list of model aliases corresponding to the model names.
    - export_name (str): The name for the evaluation file.
    - models_folder (str): The folder path where the models are stored.
    - title (str, optional): The title of the exported notebook. Defaults to None.
    
    Returns:
    - None
"""

import papermill as pm
import os
from datetime import date
import subprocess


######## Start of Input section
export_name = 'Eval_Test' + '_' + str(date.today()) + '.ipynb'
model_names = [model_name]
model_alias = ['"Test Model"']
title = 'Test evaluation'
model_folder = os.path.join('05_models','train_test')


######## End of Input section 

# Evaluation  function. Do not change anything here!!!!!
def eval_models(model_names, model_alias, export_name, models_folder, base_name = 'model_testing.ipynb', output_format = 'html', title = None):
    base_path = os.path.join(os.getcwd(), base_name)
    export_path = os.path.join(os.getcwd(), '07_model_compare',  export_name)

    res = pm.execute_notebook(base_path, export_path, parameters = dict(model_names=model_names, model_alias=model_alias, base_folder = models_folder, title = title))
    # Export the notebook to HTML format
    subprocess.run(['jupyter', 'nbconvert', '--to', output_format, '--no-input', export_path])

    return None

# Excute the evaluation function
eval_models(model_names, model_alias, export_name, model_folder, title = title)