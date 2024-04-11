"""
This script contains functions for sequencing and normalizing data.

Functions:
- sequence_data: Sequences the input and output data based on lag, delay, and prediction steps.
- sequence_sample_random: Sequences a random sample of input and output data based on lag, delay, and prediction steps.
- sequence_list: Sequences a list of input and output data based on lag, delay, and prediction steps.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def in_slicer(in_sample, lag, delay, prediction_steps):
    N = in_sample.shape[0]
    k = N - (lag + delay + prediction_steps)
    # make slicer to extract sequences from in and out data
    in_slice = np.array([range(i, i + lag) for i in range(k)])
    # slice and append data
    in_sliced = in_sample[in_slice, :]
    return in_sliced

def out_slicer(out_sample, lag, delay, prediction_steps):
    out_sample = np.reshape(out_sample, (out_sample.shape[0], 1))
    N = out_sample.shape[0]
    k = N - (lag + delay + prediction_steps)
    # make slicer to extract sequences from in and out data
    out_slice = np.array([range(i + lag + delay, i + lag + delay + prediction_steps) for i in range(k)])
    # slice and append data
    out_sliced = out_sample[out_slice, :]

    return out_sliced


def sequence_for_sequential(sims_data, in_vars=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
    """
    Sequences the input and output data based on lag, delay, and prediction steps.

    Args:
        sims_data (list): List of data samples.
        in_vars (list, optional): List of input variables. Defaults to ['duration', 'p'].
        out_vars (list, optional): List of output variables. Defaults to ['Q_out'].
        in_scaler (object, optional): Scaler object for input data. Defaults to None.
        out_scaler (object, optional): Scaler object for output data. Defaults to None.
        lag (int, optional): Number of lagged time steps to include in the input sequence. Defaults to 36.
        delay (int, optional): Number of time steps to delay the output sequence. Defaults to 0.
        prediction_steps (int, optional): Number of time steps to predict in the output sequence. Defaults to 12.

    Returns:
        List of Sequenced input data and output data.
    """
    # get output variables if not specified
    if out_vars is None:
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]

    in_data = np.array([])
    out_data = np.array([])


    for sample in sims_data:
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        # N = in_sample.shape[0]
        # k = N - (lag + delay + prediction_steps)
        # # make slicer to extract sequences from in and out data
        # in_slice = np.array([range(i, i + l) for i in range(k)])
        # out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        
        # slice and append data
        in_sliced, out_sliced = slicer(in_sample, out_sample, lag, delay, prediction_steps)
        if in_data.size == 0:
            in_data = in_sliced
            out_data = out_sliced
        else:
            in_data = np.append(in_data, in_sliced, axis=0)
            out_data = np.append(out_data, out_sliced, axis=0)

    return in_data, out_data




def sequence_sample_random(sims_data, in_vars=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, random_seed=42):
    """
    Sequences a random sample of input and output data based on lag, delay, and prediction steps.

    Args:
        sims_data (list): List of data samples.
        in_vars (list, optional): List of input variables. Defaults to ['duration', 'p'].
        out_vars (list, optional): List of output variables. Defaults to ['Q_out'].
        in_scaler (object, optional): Scaler object for input data. Defaults to None.
        out_scaler (object, optional): Scaler object for output data. Defaults to None.
        lag (int, optional): Number of lagged time steps to include in the input sequence. Defaults to 36.
        delay (int, optional): Number of time steps to delay the output sequence. Defaults to 0.
        prediction_steps (int, optional): Number of time steps to predict in the output sequence. Defaults to 12.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        List of Sequenced input data and output data.
    """
    # get output variables if not specified
    if out_vars is None:
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]

    in_data = np.array([])
    out_data = np.array([])
    
    # get random event
    rnd_indices = np.random.choice(len(sims_data))
    sample = sims_data[rnd_indices]

    # get input and output data of event
    in_sample = np.array(sample[1][in_vars])
    out_sample = np.array(sample[1][out_vars])
    # normalize data with given scalers
    in_sample = in_scaler.transform(in_sample)
    out_sample = out_scaler.transform(out_sample)

    #slice data
    in_data  = in_slicer(in_sample, lag, delay, prediction_steps)
    out_data = out_slicer(out_sample, lag, delay, prediction_steps)
    return in_data, out_data

# returns list with col 0 = dictionary of event data, col 1 = input data, col 2 = output data
def sequence_list(sims_data, in_vars=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, buffer_time = pd.Timedelta('2h')):
    """
    Sequences a list of input and output data based on lag, delay, and prediction steps.

    Args:
        sims_data (list): List of data samples.
        in_vars (list, optional): List of input variables. Defaults to ['duration', 'p'].
        out_vars (list, optional): List of output variables. Defaults to ['Q_out'].
        in_scaler (object, optional): Scaler object for input data. Defaults to None.
        out_scaler (object, optional): Scaler object for output data. Defaults to None.
        lag (int, optional): Number of lagged time steps to include in the input sequence. Defaults to 36.
        delay (int, optional): Number of time steps to delay the output sequence. Defaults to 0.
        prediction_steps (int, optional): Number of time steps to predict in the output sequence. Defaults to 12.
        buffer_time (pd.Timedelta, optional): Buffer time to subtract from event duration. Defaults to pd.Timedelta('2h').

    Returns:
     - list with sequenced data without transformation
     - list with sequenced data with transformation

    each event in each list contains:
        - col 0 = dictionary of event data
        - col 1 = input data
        - col 2 = output data
    """

    # get output variables if not specified
    if out_vars is None:
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]

    in_seq_trans = np.array([])
    out_seq_trans = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    sequenced_list_trans = []
    sequenced_list = []

    for sample in sims_data:
        sample_name = sample[0].replace('.out', '')
        intervall = sample[1].index[1] - sample[1].index[0]
        intervall = int(intervall.total_seconds() / 60)
        
        # get meta data of events
        if 'e2' in sample_name:
            type = 'Euler Typ 2'
        else:
            type = 'Aufgezeichnet'
        
        if 'duration' in sample[1]:
            event_duration = int(sample[1]['duration'].iloc[-1] - buffer_time.total_seconds() / 60)
        else:
            event_duration = None
        if 'p' in sample[1]:
            precip_sum = sample[1]['p'].sum() * intervall / 60
            max_intensity = sample[1]['p'].max()
        else:
            precip_sum = None
            max_intensity = None

        # create dictionary for event and model meta data
        # !!!!!vvvvvvvv changes here will affect other functions vvvvvvvv!!!!!!!!!!!
        meta_dictionary = {'name': sample_name, 'duration': event_duration, 'total precipitation': precip_sum, 'max intensity': max_intensity, 'interval': intervall, 'event type': type, 'lag': l, 'delay': d, 'prediction steps': n}
    	# !!!!!^^^^^^^^ changes here will affect other functions ^^^^^^^^^^!!!!!!!!!!!
     

        # append event dictionary to list
        sequenced_list.append([])
        sequenced_list[len(sequenced_list)-1].append(meta_dictionary)
        sequenced_list_trans.append([])
        sequenced_list_trans[len(sequenced_list_trans)-1].append(meta_dictionary)

        # get input and output data of event
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        # normalize data with given scalers
        in_sample_trans = in_scaler.transform(in_sample)
        out_sample_trans = out_scaler.transform(out_sample)

        #slice data
        in_seq  = in_slicer(in_sample, l, d, n)
        out_seq = out_slicer(out_sample, l, d, n)

        in_seq_trans  = in_slicer(in_sample, l, d, n)
        out_seq_trans = out_slicer(out_sample, l, d, n)

        # append data of transformed and non-transformed data to their lists
        sequenced_list[len(sequenced_list)-1].append(in_seq)
        sequenced_list[len(sequenced_list)-1].append(out_seq)
        sequenced_list_trans[len(sequenced_list_trans)-1].append(in_seq_trans)
        sequenced_list_trans[len(sequenced_list_trans)-1].append(out_seq_trans)

    return sequenced_list, sequenced_list_trans

def sequence_data(sims_data, in_vars=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
    """
    !!!! only for model using keras.Sequential
    Sequences the input and output data based on lag, delay, and prediction steps.

    Args:
        sims_data (list): List of data samples.
        in_vars (list, optional): List of input variables. Defaults to ['duration', 'p'].
        out_vars (list, optional): List of output variables. Defaults to ['Q_out'].
        in_scaler (object, optional): Scaler object for input data. Defaults to None.
        out_scaler (object, optional): Scaler object for output data. Defaults to None.
        lag (int, optional): Number of lagged time steps to include in the input sequence. Defaults to 36.
        delay (int, optional): Number of time steps to delay the output sequence. Defaults to 0.
        prediction_steps (int, optional): Number of time steps to predict in the output sequence. Defaults to 12.

    Returns:
        Sequenced input data and output data:
            in_data = input data
            out_data = output data
                - out_data itself holds a list of output sequences for each target variable
    """
    # get output variables if not specified
    if out_vars is None:
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars]

    in_data = np.array([])
    # out_data = []
    out_data = [[] for _ in range(len(out_vars))]


    for sample in sims_data:
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        # N = in_sample.shape[0]
        # k = N - (lag + delay + prediction_steps)
        # # make slicer to extract sequences from in and out data
        # in_slice = np.array([range(i, i + l) for i in range(k)])
        # out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        
        # slice and append data
        # in_sliced, out_sliced = slicer(in_sample, out_sample, lag, delay, prediction_steps)
        # if in_data.size == 0:
        #     in_data = in_sliced
        #     out_data = out_sliced
        # else:
        #     in_data = np.append(in_data, in_sliced, axis=0)
        #     out_data = np.append(out_data, out_sliced, axis=0)

        
        # slice and append data
        in_sliced  = in_slicer(in_sample, lag, delay, prediction_steps)
        
        out_sliced = []

        for target in range(len(out_vars)):
            out_sliced.append(out_slicer(out_sample[:,target], lag, delay, prediction_steps))
            if in_data.size == 0:
                out_data[target] = out_sliced[target]
            else:
                out_data[target] = np.append(out_data[target], out_sliced[target], axis=0)
                    # out_data[i] = np.append(out_data[i], out_sliced, axis=0)

        if in_data.size == 0:
            in_data = in_sliced
        else:
            in_data = np.append(in_data, in_sliced, axis=0)
       

    return in_data, out_data

if __name__ == "__main__":
    from modules.save_load_model import load_model
    import os



    model_name = 'Gievenbeck_DoubleNodeTest_LSTM_20240408'

    model_folder = os.path.join('05_models', model_name)

    model, in_scaler, out_scaler, train_data, val_data, test_data = load_model(model_folder)

    in_vars=['duration', 'p']
    out_vars = [col for col in test_data[0][1].columns if col not in in_vars]
    
    lag = int(2 * 60 / 5)
    delay = 0
    p_steps = 6

    # x_testing, y_testing = sequence_for_sequential(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
    #                                 out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    
    # print(y_testing.shape)
    # print(x_testing.shape)

    x_testing, y_testing = sequence_data(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                    out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    
    print(y_testing)
    print(x_testing.shape)
