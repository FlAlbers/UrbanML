import papermill as pm
import os
from datetime import date
import subprocess

def compare_models(model_names, model_alias, export_name, models_folder, base_name = 'model_testing.ipynb', output_format = 'html'):
    base_path = os.path.join(os.getcwd(), base_name)
    export_path = os.path.join(os.getcwd(), '07_model_compare',  export_name)

    res = pm.execute_notebook(base_path, export_path, parameters = dict(model_names=model_names, model_alias=model_alias, base_folder = models_folder))
    # Export the notebook to HTML format
    subprocess.run(['jupyter', 'nbconvert', '--to', output_format, '--no-input', export_path])

# Compare loss functions MSE, MAE, MAPE
export_name = 'Comp_Loss_functions_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-04-28', 'Gievenbeck_LSTM_Single_MAE2024-04-28', 'Gievenbeck_LSTM_Single_MAPE2024-04-28']
model_alias = ['"Loss = MSE"','"Loss = MAE"','"Loss = MAPE"']
models_folder = os.path.join('05_models', 'loss_functions_compare')
compare_models(model_names, model_alias, export_name, models_folder)


###################################################
# Compare shuffle = True and shuffle = False
export_name = 'Comp_Shuffle_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE_Shuffle_2024-04-29', 'Gievenbeck_LSTM_Single_MSE_No_Shuffle_2024-04-29']
model_alias = ['"Shuffle = True"','"Shuffle = False"']
base_folder = os.path.join('05_models', 'shuffle_compare')
compare_models(model_names, model_alias, export_name, base_folder)
