import numpy as np
import copy
from modules.sequence_and_normalize import sequence_data

def inverse_and_shape_helper(sequences, out_scaler):
    if isinstance(sequences, list):
        for i in range(len(sequences)):
            # Predict = np.append(Predict[0].reshape(Predict[0].shape[0],Predict[0].shape[1],1), Predict[1].reshape(Predict[1].shape[0],Predict[1].shape[1],1), axis=2)
            if out_scaler is not None:
                inverse_var = out_scaler[i].inverse_transform(sequences[i])
            else:
                inverse_var = sequences[i]
            inverse_var = inverse_var.reshape((len(inverse_var), len(inverse_var[0]), len(inverse_var[0][0]) if inverse_var.ndim == 3 else 1)) #####
            if i == 0:
                inverse = inverse_var
            else:
                inverse = np.append(inverse, inverse_var, axis=2)
    else:
        if out_scaler is not None:
            inverse = out_scaler.inverse_transform(sequences)
        else:
            inverse = sequences
        inverse = inverse.reshape((len(inverse), len(inverse[0]), len(inverse[0][0]) if inverse.ndim == 3 else 1))
    return inverse


def pred_all_list(model, out_scaler, event_list, event_list_trans):
    """
    Perform predictions using a given model and scaler on a list of sequences sorted by event.

    Args:
        model: The trained model used for predictions.
        out_scaler: The scaler used to transform the predictions back to their original scale.
        event_list: The list of event sequences to be predicted.
        event_list_trans: The transformed version of the event list for the model.

    Returns:
        new_list: The updated event list with predictions appended.
    """
    new_list = copy.deepcopy(event_list)
    # interval = event_list[0][0]['interval']
    # delay = event_list[0][0]['delay']
    # p_steps = event_list[0][0]['prediction steps']
    for n_sample in range(len(event_list)):
        # Predict the transformed sequences
        pred = model.predict(event_list_trans[n_sample][1], verbose=0)

        pred_inverse = inverse_and_shape_helper(pred, out_scaler)
        
        new_list[n_sample].append(pred_inverse.reshape((len(pred_inverse), len(pred_inverse[0]), 1)))

    return new_list


def pred_inverse_all(raw_data, model, in_vars, out_vars, in_scaler, out_scaler =None, lag = None, delay = None, p_steps = None, in_vars_past=None):
    """
    Perform predictions using a given model and scaler on a list of sequences sorted by event.

    Args:
        raw_data: The raw data used for sequence data preparation.
        model: The trained model used for predictions.
        in_vars: The input variables used for sequence data preparation.
        out_vars: The output variables used for sequence data preparation.
        in_scaler: The scaler used for input variable normalization.
        out_scaler: The scaler used for output variable normalization.
        lag: The lag value used for sequence data preparation.
        delay: The delay value used for sequence data preparation.
        p_steps: The number of prediction steps used for sequence data preparation.

    Returns:
        pred_inverse: An array with all prediction sequences transformed back to the original unit.
        true_inverse: An array with all true sequences transformed back to the original unit.
    """
    
    # Sequenzieren und normalisieren der Daten
    x, y = sequence_data(raw_data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                        out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps, in_vars_past=in_vars_past)

    pred = model.predict(x, verbose=0)

    pred_inverse = inverse_and_shape_helper(pred, out_scaler)

    # pred_inverse = out_scaler.inverse_transform(pred)
    if isinstance(y, list):
        for i in range(len(y)):
            y[i] = y[i].squeeze(axis=2)
    else:
        if y.shape[-1] == 1:
            y = y.squeeze()

    true_inverse = inverse_and_shape_helper(y, out_scaler)

    return true_inverse, pred_inverse

# dimensionsänderung beachten wenn lstm von dense auf sequence umgestellt wird
def pred_and_add_durIndex(model, out_scaler = None, event_list = None, event_list_trans = None):
    """
    Perform predictions using a given model and scaler on a list of sequences sorted by event
    and add the duration index to the predicted and actual values.

    Args:
        model: The trained model used for predictions.
        out_scaler: The scaler used to transform the predictions back to their original scale.
        event_list: The list of event sequences to be predicted.
        event_list_trans: The transformed version of the event list for the model.

    Returns:
        new_list: The updated event list with duration in actual values and appended predictions with duration.
    """

    new_list = copy.deepcopy(event_list)
    interval = event_list[0][0]['interval']
    delay = event_list[0][0]['delay']
    p_steps = event_list[0][0]['prediction steps']

    for n_sample in range(len(event_list)):
        
        pred = model.predict(event_list_trans[n_sample][1], verbose=0)
        # print(pred[0].shape)
        pred_inverse = inverse_and_shape_helper(pred, out_scaler)
        # print(event_list_trans[n_sample][2][0].shape)
        true = event_list_trans[n_sample][2]

        if isinstance(true, list):
            for i in range(len(true)):
                true[i] = true[i].squeeze(axis=2)
        else:
            if true.shape[-1] == 1:
                true = true.squeeze()
    
        true_inverse = inverse_and_shape_helper(true, out_scaler)

        # pred_invert = pred_invert.reshape((len(pred_invert), len(pred_invert[0]), len(pred_invert[0][0]) if pred_invert.ndim == 3 else 1))  
        # true_inverse = event_list[n_sample][2]

        for n_seq in range(len(event_list[n_sample][1])):
            # Calculate the start and end time of the sequence so that a duration column can be created
            start_time = max(event_list[n_sample][1][n_seq][:, 0]) + interval + delay * interval
            end_time = start_time + p_steps * interval
            duration_col = np.arange(start_time, end_time, interval).reshape(-1, 1)
            shape = (1, len(duration_col), 1 + len(true_inverse[n_seq][0]))
            # Add the duration column to the actual and predicted sequences and reshape to make them stackable

            true_seq_dur = np.append(duration_col, true_inverse[n_seq], axis=1).reshape(shape)
            pred_seq_dur = np.append(duration_col, pred_inverse[n_seq], axis=1).reshape(shape)
            if n_seq == 0:
                # actual_dur = np.column_stack((duration_col, actual_seq[n_seq])).reshape(shape)
                
                # if isinstance(pred_inverse,list):
                    # pred_dur = np.column_stack((duration_col,pred_inverse[n_seq])).reshape(shape)                                    #####
                true_dur = true_seq_dur
                pred_dur = pred_seq_dur
            else:
                # actual_dur = np.vstack((actual_dur, np.column_stack((duration_col, actual_seq[n_seq])).reshape(shape)))
                # pred_dur = np.vstack((pred_dur, np.column_stack((duration_col, pred_inverse[n_seq])).reshape(shape)))                #####
                true_dur = np.vstack((true_dur, true_seq_dur))
                pred_dur = np.vstack((pred_dur, pred_seq_dur))
                
        # check_shape = actual_dur.shape
        del new_list[n_sample][2]
        new_list[n_sample].append(true_dur)
        new_list[n_sample].append(pred_dur)
        del true_dur

    return new_list

# function testing area
if __name__ == '__main__':
    import pandas as pd
    from tensorflow.keras.models import model_from_json
    import joblib
    import pickle
    import os
    from sklearn.preprocessing import MinMaxScaler
    from modules.sequence_and_normalize import sequence_list
  
    # Assign all relevant paths
    model_folder = os.path.join('05_models', 'loss_functions_compare','Gievenbeck_LSTM_Single_MSE2024-04-28')
    # model_folder = '05_models\\Gievenbeck_SingleNode_LSTM_20240328'
    # model_name = "Gievenbeck_LSTM_Single_MSE2024-04-28"
    # model_path = os.path.join(model_folder, f'{model_name}.json')
    # weights_path = os.path.join(model_folder, f'{model_name}.weights.h5')
    # in_scaler_path = os.path.join(model_folder, 'in_scaler.pkl')
    # out_scaler_path = os.path.join(model_folder, 'out_scaler.pkl')
    # test_data_path = os.path.join(model_folder, 'test_data')
    # Saving model design to JSON

    from modules.save_load_model import load_model_container
    
    model_container = load_model_container(model_folder)
    # sequence data to list structure
    lag = int(2 * 60 / 5)
    delay = -12
    p_steps = 12

    in_col=['duration', 'p']
    # out_col=['Q_out']


    model = model_container['selected_model']['model']
    test_data = model_container['selected_model']['test_data']
    in_scaler = model_container['selected_model']['in_scaler']
    out_scaler = model_container['selected_model']['out_scaler']

    seq_test, seq_test_trans = sequence_list(test_data, in_vars_future=in_col, in_scaler= in_scaler ,out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    
    test_list = pred_and_add_durIndex(model, out_scaler, seq_test, seq_test_trans)
    
    test_list2 = pred_all_list(model, out_scaler, seq_test, seq_test_trans)
    
    test_list[0][3]
    test_list2[0][3]

    seq_test[0][1]
    seq_test[0][2]
    seq_test[0][3]
