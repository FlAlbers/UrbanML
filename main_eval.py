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

    return None

# Compare loss functions MSE, MAE, MAPE
export_name = 'Comp_Loss_functions_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-03', 'Gievenbeck_LSTM_Single_MAE2024-05-03', 'Gievenbeck_LSTM_Single_MAPE2024-05-03']
model_alias = ['"Loss = MSE"','"Loss = MAE"','"Loss = MAPE"']
models_folder = os.path.join('05_models', 'loss_functions_compare')
compare_models(model_names, model_alias, export_name, models_folder)

# Compare loss functions MSE, MAE
export_name = 'Comp_Loss_MSE_MAE_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-03', 'Gievenbeck_LSTM_Single_MAE2024-05-03']
model_alias = ['"Loss = MSE"','"Loss = MAE"']
models_folder = os.path.join('05_models', 'loss_functions_compare')
compare_models(model_names, model_alias, export_name, models_folder)


###################################################
# Compare shuffle = True and shuffle = False
export_name = 'Comp_Shuffle_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-03', 'Gievenbeck_LSTM_Single_MSE_No_Shuffle_2024-05-03']
model_alias = ['"Shuffle = True"','"Shuffle = False"']
base_folder = os.path.join('05_models', 'shuffle_compare')
compare_models(model_names, model_alias, export_name, base_folder)

# Compare Units = 32, Units = 64 and Units = 128
export_name = 'Comp_Units_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE_Shuffle_u32_2024-05-03', 'Gievenbeck_LSTM_Single_MSE_u64_2024-05-03', 'Gievenbeck_LSTM_Single_MSE_u128_2024-05-03', 'Gievenbeck_LSTM_Single_MSE_u256_2024-05-03', 'Gievenbeck_LSTM_Single_MSE_u512_2024-05-03']
model_alias = ['"Units = 32"','"Units = 64"','"Units = 128"', '"Units = 256"', '"Units = 512"']
base_folder = os.path.join('05_models', 'units_compare')
compare_models(model_names, model_alias, export_name, base_folder)


# Compare Shallow and Deep
export_name = 'Comp_Deep_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE_u128_2024-05-03', 'Gievenbeck_LSTM_Double_MSE_u128_2024-05-03', 'Gievenbeck_LSTM_Triple_MSE_u128_2024-05-03']
model_alias = ['"1 Schicht"','"2 Schichten"','"3 Schichten"']
base_folder = os.path.join('05_models', 'deep_compare')
compare_models(model_names, model_alias, export_name, base_folder)

# Compare Shallow and Deep with different units
export_name = 'Comp_Deep_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE_u128_2024-05-03', 'Gievenbeck_LSTM_Double_MSE_u64_2024-05-04', 'Gievenbeck_LSTM_Triple_MSE_u32_2024-05-04']
model_alias = ['"1 Schicht"','"2 Schichten"','"3 Schichten"']
base_folder = os.path.join('05_models', 'deep_compare')
compare_models(model_names, model_alias, export_name, base_folder)

# Compare Test
export_name = 'Comp_add_past' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_Past_double_u128','Gievenbeck_LSTM_Single_MSE_u128_2024-05-03']
model_alias = ['"Mit vorh. Q"','"Ohne vorh. Q"']
base_folder = os.path.join('05_models', 'add_past_compare')
compare_models(model_names, model_alias, export_name, base_folder)

