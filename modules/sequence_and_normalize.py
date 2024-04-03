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


def sequence_data(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
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
    
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    for sample in sims_data:
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        N = in_sample.shape[0]
        k = N - (lag + delay + prediction_steps)
        # make slicer to extract sequences from in and out data
        in_slice = np.array([range(i, i + l) for i in range(k)])
        out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        # slice and append data
        if in_data.size == 0:
            in_data = in_sample[in_slice, :]
            out_data = out_sample[out_slice, :]
        else:
            in_data = np.append(in_data, in_sample[in_slice, :], axis=0)
            out_data = np.append(out_data, out_sample[out_slice, :], axis=0)
    return in_data, out_data

def sequence_sample_random(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, random_seed=42):
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
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    rnd_indices = np.random.choice(len(sims_data))
    sample = sims_data[rnd_indices]
    in_sample = np.array(sample[1][in_vars])
    out_sample = np.array(sample[1][out_vars])
    in_sample = in_scaler.transform(in_sample)
    out_sample = out_scaler.transform(out_sample)

    N = in_sample.shape[0]
    k = N - (lag + delay + prediction_steps)

    # make slicer to extract sequences from in and out data
    in_slice = np.array([range(i, i + l) for i in range(k)])
    out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

    # slice and append data
    in_data = in_sample[in_slice, :]
    out_data = out_sample[out_slice, :]
    return in_data, out_data

# returns list with col 0 = dictionary of event data, col 1 = input data, col 2 = output data
def sequence_list(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12, buffer_time = pd.Timedelta('2h')):
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
        List containing Data sorted for each event with dictionary of event data, sequenced input data and sequenced output data.
    """
    
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    sequenced_list = []

    for sample in sims_data:
        sample_name = sample[0].replace('.out', '')
        intervall = sample[1].index[1] - sample[1].index[0]
        intervall = int(intervall.total_seconds() / 60)
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

        event_dict = {'name': sample_name, 'duration': event_duration, 'total precipitation': precip_sum, 'max intensity': max_intensity, 'intervall': intervall}

        sequenced_list.append([])
        sequenced_list[len(sequenced_list)-1].append(event_dict)
        in_sample = np.array(sample[1][in_vars])
        out_sample = np.array(sample[1][out_vars])
        in_sample = in_scaler.transform(in_sample)
        out_sample = out_scaler.transform(out_sample)

        N = in_sample.shape[0]
        k = N - (lag + delay + prediction_steps)
        
        # make slicer to extract sequences from in and out data
        in_slice = np.array([range(i, i + l) for i in range(k)])
        out_slice = np.array([range(i + l + d, i + l + d + n) for i in range(k)])

        # slice and append data
        in_data = in_sample[in_slice, :]
        out_data = out_sample[out_slice, :]

        sequenced_list[len(sequenced_list)-1].append(in_data)
        sequenced_list[len(sequenced_list)-1].append(out_data)

    # returns list with col 0 = dictionary of event data, col 1 = input data, col 2 = output data
    return sequenced_list