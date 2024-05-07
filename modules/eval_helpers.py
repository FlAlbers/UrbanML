
import numpy as np
from modules.sequence_and_normalize import sequence_data
from modules.save_load_model import load_model
from sklearn.metrics import mean_squared_error
import math

def mae_mse_rmse(true, pred):
    mse = np.square(np.subtract(true,pred)).mean()
    rmse = math.sqrt(mse)
    mae = abs(np.subtract(true,pred)).mean()
    eval_dict = {'mae': mae, 'mse': mse, 'rmse': rmse}
    return eval_dict

def rmse_from_raw(raw_data, model, in_vars, out_vars, in_scaler, out_scaler, lag, delay, p_steps):
    """
    Calculate the root mean squared error (RMSE) for a given model and one or multiple event datasets.

    Args:
        raw_data (list or ndarray): The raw data or list of raw data to be used for evaluation.
        model: The trained model used for prediction.
        in_vars (list): The input variables used for prediction.
        out_vars (list): The output variables to be predicted.
        in_scaler: The scaler used to normalize the input variables.
        out_scaler: The scaler used to normalize the output variables.
        lag (int): The number of previous time steps to consider as input.
        delay (int): The number of future time steps to predict.
        p_steps (int): The number of prediction steps to make.

    Returns:
        ndarray: An array of RMSE values for each prediction made.

    Raises:
        None

    """
    if isinstance(raw_data,list):
        rmse_all = ()
        for data in raw_data:
            x, y = sequence_data(data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                            out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

            pred = model.predict(x, verbose=1)
            pred_inverse = out_scaler.inverse_transform(pred)
            if y.shape[-1] == 1:
                y = y.squeeze()
            true_inverse = out_scaler.inverse_transform(y)
            mse = np.square(np.subtract(true_inverse,pred_inverse)).mean()
            rmse = math.sqrt(mse)
            rmse_all = np.append(rmse_all, rmse)
    else:    
        x, y = sequence_data(raw_data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                            out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

        pred = model.predict(x, verbose=1)
        pred_inverse = out_scaler.inverse_transform(pred)
        if y.shape[-1] == 1:
                y = y.squeeze()
        true_inverse = out_scaler.inverse_transform(y)
        mse = np.square(np.subtract(true_inverse,pred_inverse)).mean()
        rmse = np.sqrt(mse)
        rmse_all = np.append(rmse_all, rmse)

    return rmse_all

def mae_from_raw(raw_data, model, in_vars, out_vars, in_scaler, out_scaler, lag, delay, p_steps):
    """
    Calculate the root mean squared error (RMSE) for a given model and one or multiple event datasets.

    Args:
        raw_data (list or ndarray): The raw data or list of raw data to be used for evaluation.
        model: The trained model used for prediction.
        in_vars (list): The input variables used for prediction.
        out_vars (list): The output variables to be predicted.
        in_scaler: The scaler used to normalize the input variables.
        out_scaler: The scaler used to normalize the output variables.
        lag (int): The number of previous time steps to consider as input.
        delay (int): The number of future time steps to predict.
        p_steps (int): The number of prediction steps to make.

    Returns:
        ndarray: An array of RMSE values for each prediction made.

    Raises:
        None

    """
    if isinstance(raw_data,list):
        mae_all = ()
        for data in raw_data:
            x, y = sequence_data(data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                            out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

            pred = model.predict(x, verbose=1)
            pred_inverse = out_scaler.inverse_transform(pred)
            if y.shape[-1] == 1:
                y = y.squeeze()
            true_inverse = out_scaler.inverse_transform(y)
            mae = np.subtract(true_inverse,pred_inverse).mean()
            mae_all = np.append(mae_all, mae)
    else:    
        x, y = sequence_data(raw_data, in_vars_future=in_vars, out_vars=out_vars, in_scaler=in_scaler, 
                                            out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps)

        pred = model.predict(x, verbose=1)
        pred_inverse = out_scaler.inverse_transform(pred)
        if y.shape[-1] == 1:
                y = y.squeeze()
        true_inverse = out_scaler.inverse_transform(y)
        mae = np.subtract(true_inverse,pred_inverse).mean()
        mae_all = np.append(mae_all, mae)

    return mae_all



if __name__ == '__main__':
    import os
    # Load the model
    model_name = 'Gievenbeck_LSTM_Single_1h_P_20240408'
    model_folder = os.path.join('05_models', model_name)
    # model_folder = os.path.join('05_models', 'Gievenbeck_SingleNode_LSTM_20240328')
    model, in_scaler, out_scaler, train_data, val_data, test_data, data_info_dict = load_model(model_folder)
    lag = data_info_dict['lag']
    delay = data_info_dict['delay']
    p_steps = data_info_dict['prediction_steps']

    in_vars= data_info_dict['in_vars']
    out_vars= data_info_dict['out_vars']
    rmse = rmse_from_raw([train_data, val_data, test_data], model, in_vars, out_vars, in_scaler, out_scaler, lag, delay, p_steps)