"""
Author: Flemming Albers

This script contains functions for sequencing and normalizing data.

Functions:
- sequence_data: Sequences the input and output data based on lag, delay, and prediction steps.
- sequence_sample_random: Sequences a random sample of input and output data based on lag, delay, and prediction steps.
- sequence_list: Sequences a list of input and output data based on lag, delay, and prediction steps.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# slicer for input data that include future values
def in_slicer_future(in_sample, lag, delay, prediction_steps):
    N = in_sample.shape[0]
    k = N - (lag + delay + prediction_steps)
    # make slicer to extract sequences from in and out data
    in_slice = np.array([range(i, i + lag) for i in range(k)])
    # slice and append data
    in_sliced = in_sample[in_slice, :]
    return in_sliced

# slicer for input data that that does NOT include future values
def in_slicer_past(in_event_past, lag, delay, prediction_steps):
    in_pre_slice = in_slicer_future(in_event_past, lag, delay, prediction_steps)
    for i in range(len(in_pre_slice)):
        range_past = range((lag + delay),lag)
        in_pre_slice[i,range_past,0] = 0
    return in_pre_slice

# slicer for output data
def out_slicer(out_sample, lag, delay, prediction_steps):
    out_sample = np.reshape(out_sample, (out_sample.shape[0], 1))
    N = out_sample.shape[0]
    k = N - (lag + delay + prediction_steps)
    # make slicer to extract sequences from in and out data
    out_slice = np.array([range(i + lag + delay, i + lag + delay + prediction_steps) for i in range(k)])
    # slice and append data
    out_sliced = out_sample[out_slice, :]

    return out_sliced



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
    in_data  = in_slicer_future(in_sample, lag, delay, prediction_steps)
    out_data = out_slicer(out_sample, lag, delay, prediction_steps)
    return in_data, out_data

def sequence_data(sims_data, in_vars_future=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, in_vars_past = None):
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
        incl_prev (bool, optional): If True, input data also includes previous output data. Defaults to False.

    Returns:
        Sequenced input data and output data:
            in_data = input data
            out_data = output data
                - out_data itself holds a list of output sequences for each target variable
    """
    # get output variables if not specified
    if out_vars is None:
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars_future]

    if not isinstance(out_vars, list):
        out_vars = [out_vars]

    # include output columns in input data if incl_prev is True
    if in_vars_past is not None:
        in_vars = in_vars_future + in_vars_past
    else:
        in_vars = in_vars_future

    in_data = np.array([])
    # out_data = []
    out_data = [[] for _ in range(len(out_vars))]


    for event in sims_data:
        # in_event_fut = np.array(event[1][in_vars_future])
        in_event = np.array(event[1][in_vars])
        out_event = np.array(event[1][out_vars])

        if in_scaler is not None:
            in_event = in_scaler.transform(in_event)
        
        in_event_future = in_event[:,range(len(in_vars_future))]
        in_event_past = in_event[:,range(len(in_vars_future), len(in_vars))]

        if out_scaler is not None:
            out_event = out_scaler.transform(out_event)

        out_sliced = []

        # slice and append oputput data
        if len(out_vars) > 1:
            for target in range(len(out_vars)):
                out_sliced.append(out_slicer(out_event[:,target], lag, delay, prediction_steps))
                if in_data.size == 0:
                    out_data[target] = out_sliced[target]
                else:
                    out_data[target] = np.append(out_data[target], out_sliced[target], axis=0)
        else:
            out_sliced = out_slicer(out_event, lag, delay, prediction_steps)
            if in_data.size == 0:
                out_data = out_sliced
            else:
                out_data = np.append(out_data, out_sliced, axis=0)

        
        # slice and append input data
        in_sliced_future  = in_slicer_future(in_event_future, lag, delay, prediction_steps)
        if in_vars_past is not None:
            in_sliced_past = in_slicer_past(in_event_past, lag, delay, prediction_steps)
            in_sliced = np.append(in_sliced_future, in_sliced_past, axis=2)
        else:
            in_sliced = in_sliced_future

        if in_data.size == 0:
            in_data = in_sliced
        else:
            in_data = np.append(in_data, in_sliced, axis=0)

    return in_data, out_data

# returns list with col 0 = dictionary of event data, col 1 = input data, col 2 = output data
def sequence_list(sims_data, in_vars_future=['duration', 'p'], out_vars=None, in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, buffer_time = pd.Timedelta('2h'), in_vars_past=None):
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
        out_vars = [col for col in sims_data[0][1].columns if col not in in_vars_future]

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
            # Filter the DataFrame
            p_greater_0 = sample[1][sample[1]['p'] > 0]
            # Get the index of the last entry with a value > 0            
            first_index = p_greater_0['p'].index[0]
            last_index = p_greater_0['p'].index[-1]
            event_duration = int((last_index - first_index).total_seconds() / 60)
            # event_duration = int(sample[1]['duration'].iloc[-1] - buffer_time.total_seconds() / 60 * 2 + intervall*2)
        else:
            event_duration = None
        if 'p' in sample[1]:
            precip_sum = sample[1]['p'].sum() * intervall / 60
            max_intensity = sample[1]['p'].max()
        else:
            precip_sum = None
            max_intensity = None


        meta_dictionary = {'name': sample_name, 'duration': event_duration, 'total precipitation': precip_sum, 'max intensity': max_intensity, 'interval': intervall, 'event type': type, 'lag': l, 'delay': d, 'prediction steps': n}
     
        # append event dictionary to list
        sequenced_list.append([])
        sequenced_list[len(sequenced_list)-1].append(meta_dictionary)
        sequenced_list_trans.append([])
        sequenced_list_trans[len(sequenced_list_trans)-1].append(meta_dictionary)

        # # get input and output data of event
        # in_sample = np.array(sample[1][in_vars])
        # out_sample = np.array(sample[1][out_vars])
        # # normalize data with given scalers
        # in_sample_trans = in_scaler.transform(in_sample)
        # out_sample_trans = out_scaler.transform(out_sample)

        # #slice data
        # in_seq  = in_slicer(in_sample, l, d, n)
        # out_seq = out_slicer(out_sample, l, d, n)

        # in_seq_trans  = in_slicer(in_sample, l, d, n)
        # out_seq_trans = out_slicer(out_sample, l, d, n)

        in_seq, out_seq = sequence_data([sample], in_vars_future=in_vars_future, in_vars_past=in_vars_past, out_vars=out_vars, in_scaler=None, out_scaler=None, lag=lag, delay=delay, prediction_steps=prediction_steps)
        in_seq_trans, out_seq_trans = sequence_data([sample], in_vars_future=in_vars_future, in_vars_past=in_vars_past, out_vars=out_vars, in_scaler=in_scaler, out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=prediction_steps)

        # append data of transformed and non-transformed data to their lists
        sequenced_list[len(sequenced_list)-1].append(in_seq)
        sequenced_list[len(sequenced_list)-1].append(out_seq)
        sequenced_list_trans[len(sequenced_list_trans)-1].append(in_seq_trans)
        sequenced_list_trans[len(sequenced_list_trans)-1].append(out_seq_trans)

    return sequenced_list, sequenced_list_trans


if __name__ == "__main__":
    from modules.save_load_model import load_model, load_model_container
    import os

    model_name = 'Gievenbeck_LSTM_Single_MSE_u128_2024-05-03'

    model_folder = os.path.join('05_models', 'units_compare', model_name)

    model_container  = load_model_container(model_folder)

    test_data = model_container['model_0']['test_data']
    in_vars=['duration', 'p']
    out_vars = [col for col in test_data[0][1].columns if col not in in_vars]
    
    lag = int(2 * 60 / 5)
    delay = -12
    p_steps = 12

    # x_testing_seq, y_testing_seq = sequence_for_sequential(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
    #                                 out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    # print(y_testing_seq.shape)    
    # print(x_testing_seq.shape)


    # x_testing, y_testing = sequence_data(train_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
    #                                 out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

    # print(y_testing.shape)
    # print(x_testing.shape)
    x, y = sequence_data(test_data, in_vars_future=in_vars, lag=lag, delay=delay, prediction_steps=p_steps, in_vars_past=['R0019769'])

    print(y[50], x[50])
    seq_test, seq_test_trans = sequence_list(test_data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                    out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)


    # Check if the data of single and multi sequencing function is the same
    print('Singel', y_testing_seq[0])
    print('Multi', y_testing[0][0])

