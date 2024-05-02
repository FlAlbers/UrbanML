import numpy as np
import copy
from modules.sequence_and_normalize import sequence_data

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
    interval = event_list[0][0]['interval']
    delay = event_list[0][0]['delay']
    p_steps = event_list[0][0]['prediction steps']
    for n_sample in range(len(event_list)):
        # Predict the transformed sequences
        Predict = model.predict(event_list_trans[n_sample][1], verbose=0)
        Predict_invert = out_scaler.inverse_transform(Predict)
        new_list[n_sample].append(Predict_invert.reshape((len(Predict_invert), len(Predict_invert[0]), 1)))

    return new_list


def pred_inverse_all(raw_data, model, in_vars, out_vars, in_scaler, out_scaler, lag, delay, p_steps):
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
    x, y = sequence_data(raw_data, in_vars=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                            out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)


    pred = model.predict(x, verbose=0)
    pred_inverse = out_scaler.inverse_transform(pred)
    if y.shape[-1] == 1:
        y = y.squeeze()
    true_inverse = out_scaler.inverse_transform(y)

    return true_inverse, pred_inverse

# dimensionsänderung beachten wenn lstm von dense auf sequence umgestellt wird
def pred_and_add_durIndex(model, out_scaler, event_list, event_list_trans):
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
        
        Predict = model.predict(event_list_trans[n_sample][1], verbose=0)
        Predict_invert = out_scaler.inverse_transform(Predict)
        Predict_invert = Predict_invert.reshape((len(Predict_invert), len(Predict_invert[0]), len(Predict_invert[0][0]) if Predict_invert.ndim == 3 else 1))
        actual_seq = event_list[n_sample][2]

        for n in range(len(event_list[n_sample][1])):
            # Calculate the start and end time of the sequence so that a duration column can be created
            start_time = max(event_list[n_sample][1][n][:, 0]) + interval + delay * interval
            end_time = start_time + p_steps * interval
            duration_col = np.arange(start_time, end_time, interval)
            shape = (1, len(duration_col), 1 + len(actual_seq[n][0]))
            if n == 0:
                actual_dur = np.column_stack((duration_col, actual_seq[n])).reshape(shape)
                pred_dur = np.column_stack((duration_col,Predict_invert[n])).reshape(shape)
            else:
                actual_dur = np.vstack((actual_dur, np.column_stack((duration_col, actual_seq[n])).reshape(shape)))
                pred_dur = np.vstack((pred_dur, np.column_stack((duration_col, Predict_invert[n])).reshape(shape)))
                
        # check_shape = actual_dur.shape
        del new_list[n_sample][2]
        new_list[n_sample].append(actual_dur)
        new_list[n_sample].append(pred_dur)
        del actual_dur

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

    seq_test, seq_test_trans = sequence_list(test_data, in_vars=in_col, in_scaler= in_scaler ,out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)
    
    test_list = pred_and_add_durIndex(model, out_scaler, seq_test, seq_test_trans)
    
    test_list2 = pred_all_list(model, out_scaler, seq_test, seq_test_trans)
    
    test_list[0][3]
    test_list2[0][3]

    seq_test[0][1]
    seq_test[0][2]
    seq_test[0][3]
