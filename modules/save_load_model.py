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

def save_model(model, save_folder, in_scaler, out_scaler, test_data):
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
    test_data_path = os.path.join(save_folder, 'test_data')

    # Saving model design to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # Saving weights to HDF5
    model.save_weights(weights_path)

    # Save the scalers
    joblib.dump(in_scaler, in_scaler_path)
    joblib.dump(out_scaler, out_scaler_path)

    # Save test data
    with open(test_data_path, 'wb') as file:
        pickle.dump(test_data, file)
    print("Saved model, scaler, and test data to disk")

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
    """
    # Assign all relevant paths
    model_path = os.path.join(model_folder, 'model.json')
    weights_path = os.path.join(model_folder, 'model.weights.h5')
    in_scaler_path = os.path.join(model_folder, 'in_scaler.pkl')
    out_scaler_path = os.path.join(model_folder, 'out_scaler.pkl')
    test_data_path = os.path.join(model_folder, 'test_data')
        
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

    # Load the test data
    with open(test_data_path, 'rb') as file:
        test_data = pickle.load(file)

    print("Loaded model from disk")

    return model, in_scaler, out_scaler, test_data

# Test Area for functions
if __name__ == '__main__':
    model_name = 'Gievenbeck_DoubleNodeTest_LSTM_20240408'
    model_folder = os.path.join('05_models', model_name)
    model, in_scaler, out_scaler, test_data = load_model(model_folder)

    model.summary()