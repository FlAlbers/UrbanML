import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def sequence_data(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
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

def sequence_list(sims_data, in_vars=['duration', 'p'], out_vars=['Q_out'], in_scaler=None, out_scaler=None, lag = 36, delay = 0, prediction_steps = 12):
    in_data = np.array([])
    out_data = np.array([])
    l = lag
    d = delay
    n = prediction_steps

    sequenced_list = []

    for sample in sims_data:
        sequenced_list.append([])
        sequenced_list[len(sequenced_list)-1].append(sample[0])
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

    return sequenced_list