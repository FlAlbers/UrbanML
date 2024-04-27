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
import csv
import pandas as pd
import matplotlib.pyplot as plt

def save_model(model_name = None, model =None, save_folder = None, in_scaler = None, out_scaler = None, train_data=None, val_data = None, test_data=None, lag =None, delay=None, prediction_steps=None, seed_train_val_test = None, seed_train_val=None, in_vars=None, out_vars=None, cv_scores=None, cv_models=None, history = None):
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
    
    

    # Create data_info dictionary
    data_info_dict = {
        'model_name': model_name,
        'lag': lag,
        'delay': delay,
        'prediction_steps': prediction_steps,
        'seed_train_val_test': seed_train_val_test,
        'seed_train_val': seed_train_val,
        'in_vars': in_vars,
        'out_vars': out_vars
    }

    # Save data_info with pickle
    data_info_path = os.path.join(save_folder, 'data_info_dict.pkl')
    with open(data_info_path, 'wb') as file:
        pickle.dump(data_info_dict, file)

    # Save cv_scores as CSV
    cv_scores_path = os.path.join(save_folder, 'cv_scores.csv')
    cv_scores.to_csv(cv_scores_path, index=True, header=True)

    # Save cv_models with pickle
    cv_models_path = os.path.join(save_folder, 'cv_models.pkl')
    with open(cv_models_path, 'wb') as file:
        pickle.dump(cv_models, file)
    
    history_path = os.path.join(model_folder, 'history.pkl')
    with open(history_path, 'wb') as file:
        pickle.dump(history, file)
    
    print("Saved model to disk")

    return None


def load_model(model_folder, print_info = True):
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
    cv_scores_path = os.path.join(model_folder, 'cv_scores.csv')
    cv_models_path = os.path.join(model_folder, 'cv_models.pkl')
    history_path = os.path.join(model_folder, 'history.pkl')

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
    
    if os.path.exists(cv_scores_path):
        cv_scores = pd.read_csv(cv_scores_path)
    else:
        cv_scores = 'unknown'

    if os.path.exists(cv_models_path):
        with open(cv_models_path, 'rb') as file:
            cv_models = pickle.load(file)
    else:
        cv_models = 'unknown'

    if os.path.exists(history_path):
        with open(history_path, 'rb') as file:
            history = pickle.load(file)
    else:
        history = 'unknown'
    

    if print_info == True:
        print("Loaded model from disk")

    return model, in_scaler, out_scaler, train_data, validation_data, test_data, data_info_dict, cv_scores, cv_models, history

def load_model_containerOLD(model_folder, print_info = True):
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
    model, in_scaler, out_scaler, train_data, validation_data, test_data, data_info_dict, cv_scores, cv_models, history = load_model(model_folder, print_info)

    model_container = {
        'name' : data_info_dict['model_name'],
        'model': model,
        'in_scaler': in_scaler,
        'out_scaler': out_scaler,
        'train_data': train_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'lag': data_info_dict['lag'],
        'delay': data_info_dict['delay'],
        'prediction_steps': data_info_dict['prediction_steps'],
        'seed_train_val_test': data_info_dict['seed_train_val_test'],
        'seed_train_val': data_info_dict['seed_train_val'],
        'in_vars': data_info_dict['in_vars'],
        'out_vars': data_info_dict['out_vars'],
        'history': history,
        'cv_scores': cv_scores,
        'cv_models': cv_models
    }

    return model_container

def save_model_containerOLD(model_container, save_folder = None):
    model_name = model_container['name']
    model = model_container['model']
    in_scaler = model_container['in_scaler']
    out_scaler = model_container['out_scaler']
    train_data = model_container['train_data']
    val_data = model_container['validation_data']
    test_data = model_container['test_data']
    lag = model_container['lag']
    delay = model_container['delay']
    prediction_steps = model_container['prediction_steps']
    seed_train_val_test = model_container['seed_train_val_test']
    seed_train_val = model_container['seed_train_val']
    in_vars = model_container['in_vars']
    out_vars = model_container['out_vars']
    cv_scores = model_container['cv_scores']
    cv_models = model_container['cv_models']
    history = model_container['history']

    save_model(model_name = model_name, model =model, save_folder = save_folder, in_scaler = in_scaler, 
               out_scaler = out_scaler, train_data=train_data, val_data = val_data, test_data=test_data, lag =lag, 
               delay=delay, prediction_steps=prediction_steps, seed_train_val_test = seed_train_val_test, 
               seed_train_val=seed_train_val, in_vars=in_vars, out_vars=out_vars, cv_scores=cv_scores, cv_models=cv_models, history = history)
    
    return None
    
def load_model_container(save_folder, print_info = True):
    """
    Loads the model container from disk.
    Contains:
        - the model, input scaler, output scaler, test data, lag, delay, prediction_steps, 
        - seed_train_val_test, seed_train_val, in_vars, out_vars, history, cv_scores, cv_models.
    
    Args:
        model_folder (str): The folder name where the model and related files are saved. (Only the relative path in the project folder is needed)
        print_info (bool): If True, prints "Loaded model from disk".

    """
    container_path = os.path.join(save_folder, 'model_dict.pkl')
    if os.path.exists(container_path):
        with open(container_path, 'rb') as file:
            model_container = pickle.load(file)
    else:
        model_container = None

    if print_info == True:
        print("Loaded model from disk")

    i=0
    for file in os.listdir(save_folder):
        if file.endswith('.json'):
            i += 1

    for i in range(i):
        model_save_name = 'model_' + str(i) + '.json'
        model_path = os.path.join(save_folder, model_save_name)
        weights_save_name = 'model_' + str(i) + '.weights.h5'
        weights_path = os.path.join(save_folder, weights_save_name)
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(weights_path)
        m_name = 'model_' + str(i)
        model_container[m_name]['model'] = loaded_model

    return model_container


def save_model_container(model_container, save_folder = None):
    # Create model_folder if not existing
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    container_path = os.path.join(save_folder, 'model_dict.pkl')
    i = 0
    for key in model_container.keys():
        if key.startswith('model'):
            
            model_save_name = 'model_' + str(i) + '.json'
            model_path = os.path.join(save_folder, model_save_name)
            weights_save_name = 'model_' + str(i) + '.weights.h5'
            weights_path = os.path.join(save_folder, weights_save_name)
            model_json = model_container[key]['model'].to_json()
            with open(model_path, "w") as json_file:
                json_file.write(model_json)
            # Saving weights to HDF5
            model_container[key]['model'].save_weights(weights_path)
            i+= 1

    # Save the models temporarily and delete them from the model_container
    models = []
    for key in model_container.keys():
        if key.startswith('model'):
            models.append(model_container[key]['model'])
            del model_container[key]['model']

    # Save the model container without the models
    with open(container_path, 'wb') as file:
        pickle.dump(model_container, file)

    # Save the models back to the model_container
    i = 0
    for key in model_container.keys():
        if key.startswith('model'):
            model_container[key]['model'] = models[i]
            i += 1
    # plt.plot(model_container['history']['loss'], '--', label='Training')
    # plt.plot(model_container['history']['val_loss'], label='Validierung')
    # plt.xlabel('Trainingsepoche')
    # plt.ylabel('Mittlerer quadratischer Fehler [-]')
    # plt.legend()
    # figure_path = os.path.join(save_folder, 'learning_curve.png')
    # plt.savefig(model_container)
    return None




# Test Area for functions
if __name__ == '__main__':

    model_name = 'Gievenbeck_LSTM_Single_Shuffle_CV_1h_P_20240408'
    model_folder = os.path.join('05_models', model_name)
    # model, in_scaler, out_scaler, test_data = load_model(model_folder)

    model_container = load_model_container(model_folder)
    
    save_model_container(model_container, model_folder)
    model_container['model_4'] = model_container.pop('model5')
    # model_container = load_model_containerOLD(model_folder)
    # save_model_container(model_container, save_folder = model_folder)
    # save_folder = model_folder
# model_container['model_2'].keys()
    # model.summary()