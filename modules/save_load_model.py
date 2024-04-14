'''
Author: Flemming Albers

Functions for loading and saving a model.
The model is saved as a JSON file and the weights are saved as a HDF5 file.
The scalers are saved as pickle files.
'''

import os
from keras.models import Model, model_from_json
import joblib
import os
import pickle

def save_model(model, save_folder, in_scaler, out_scaler, train_data=None, val_data = None, test_data=None, lag =None, delay=None, prediction_steps=None, random_seed=None, in_vars=None, out_vars=None):
    """
    Saves the model, input scaler, output scaler, and test data to disk.

    Args:
        model (keras.models.Model): The trained model to be saved.
        save_folder (str): The folder path where the model and related files will be saved.
        in_scaler: The input scaler used for preprocessing.
        out_scaler: The output scaler used for preprocessing.
        test_data: The test data used for evaluation.

    Returns:
        None
    """
    # Create model_folder if not existing
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Assign all relevant paths
    model_path = os.path.join(save_folder, 'model.json')
    weights_path = os.path.join(save_folder, 'model.weights.h5')
    in_scaler_path = os.path.join(save_folder, 'in_scaler.pkl')
    out_scaler_path = os.path.join(save_folder, 'out_scaler.pkl')
    train_data_path = os.path.join(save_folder, 'train_data')
    validation_data_path = os.path.join(save_folder, 'validation_data')
    test_data_path = os.path.join(save_folder, 'test_data')
    data_paths = [(train_data_path, train_data), (validation_data_path, val_data), (test_data_path, test_data)]

    # Saving model design to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # Saving weights to HDF5
    model.save_weights(weights_path)

    # Save the scalers
    joblib.dump(in_scaler, in_scaler_path)
    joblib.dump(out_scaler, out_scaler_path)

    # Save train, validation and test data
    for data in data_paths:
        if data[1] is not None:
            with open(data[0], 'wb') as file:
                pickle.dump(data[1], file)
    
    print("Saved model, scaler, and test data to disk")

    # Set default value for lag if None
    if lag is None:
        lag = 'unknown'
    if delay is None:
        delay = 'unknown'    
    if prediction_steps is None:
        prediction_steps = 'unknown'
    if random_seed is None:
        random_seed = 'unknown'
    if in_vars is None:
        in_vars = 'unknown'
    if out_vars is None:
        out_vars = 'unknown'


    # Create data_info dictionary
    data_info_dict = {
        'lag': lag,
        'delay': delay,
        'prediction_steps': prediction_steps,
        'random_seed': random_seed,
        'in_vars': in_vars,
        'out_vars': out_vars
    }

    # Save data_info with pickle
    data_info_path = os.path.join(save_folder, 'data_info_dict.pkl')
    with open(data_info_path, 'wb') as file:
        pickle.dump(data_info_dict, file)

    return None


def load_model(model_folder):
    """
    Loads the model, input scaler, output scaler, and test data from disk.
    
    Args:
        model_folder (str): The folder path where the model and related files are saved.

    Returns:
        model (keras.models.Model): The loaded model.
        in_scaler: The loaded input scaler.
        out_scaler: The loaded output scaler.
        test_data: The loaded test data.

    Use like this:
        model, in_scaler, out_scaler, train_data, val_data, test_data, data_info_dict = load_model(model_folder)
    """
    # Assign all relevant paths
    model_path = os.path.join(model_folder, 'model.json')
    weights_path = os.path.join(model_folder, 'model.weights.h5')
    in_scaler_path = os.path.join(model_folder, 'in_scaler.pkl')
    out_scaler_path = os.path.join(model_folder, 'out_scaler.pkl')
    train_data_path = os.path.join(model_folder, 'train_data')
    validation_data_path = os.path.join(model_folder, 'validation_data')
    test_data_path = os.path.join(model_folder, 'test_data')
    data_info_path = os.path.join(model_folder, 'data_info_dict.pkl')

    # Load the model and the scalers
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_path)

    # loaded_model.summary()

    # Load the scalers
    in_scaler = joblib.load(in_scaler_path)
    out_scaler = joblib.load(out_scaler_path)

    # Load the train, validation, test data
    if os.path.exists(train_data_path):
        with open(train_data_path, 'rb') as file:
            train_data = pickle.load(file)
    else:
        train_data = 'unknown'
    if os.path.exists(validation_data_path):
        with open(validation_data_path, 'rb') as file:
            validation_data = pickle.load(file)
    else:
        validation_data = 'unknown'
    if os.path.exists(test_data_path):
        with open(test_data_path, 'rb') as file:
            test_data = pickle.load(file)
    else:
        test_data = 'unknown'
    if os.path.exists(data_info_path):
        with open(data_info_path, 'rb') as file:
            data_info_dict = pickle.load(file)
    else:
        data_info_dict = 'unknown'
    

    print("Loaded model from disk")

    return model, in_scaler, out_scaler, train_data, validation_data, test_data, data_info_dict

# Test Area for functions
if __name__ == '__main__':
    model_name = 'Gievenbeck_DoubleNodeTest_LSTM_20240408'
    model_folder = os.path.join('05_models', model_name)
    model, in_scaler, out_scaler, test_data = load_model(model_folder)
    save_folder = model_folder

    model.summary()