import papermill as pm
import os
from datetime import date
import subprocess

def compare_models(model_names, model_alias, export_name, models_folder, base_name = 'model_testing.ipynb', output_format = 'html', title = None):
    base_path = os.path.join(os.getcwd(), base_name)
    export_path = os.path.join(os.getcwd(), '07_model_compare',  export_name)

    res = pm.execute_notebook(base_path, export_path, parameters = dict(model_names=model_names, model_alias=model_alias, base_folder = models_folder, title = title))
    # Export the notebook to HTML format
    subprocess.run(['jupyter', 'nbconvert', '--to', output_format, '--no-input', export_path])

    return None

# Test RKB
export_name = 'Test_RKB' + '_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_RKB_LSTM_2024-05-30']
model_alias = ['"RKB BÜ"']
base_folder = os.path.join('05_models', 'test_RKB')
title = 'Test RKB'
compare_models(model_names, model_alias, export_name, base_folder, title = title)


# Compare loss functions MSE, MAE, MAPE
# export_name = 'Comp_Loss_functions_' + str(date.today()) + '.ipynb'
# model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-14', 'Gievenbeck_LSTM_Single_MAE2024-05-14', 'Gievenbeck_LSTM_Single_MAPE2024-05-14']
# model_alias = ['"Loss = MSE"','"Loss = MAE"','"Loss = MAPE"']
# base_folder = os.path.join('05_models', 'loss_functions_compare')
# title = 'Vergleich Verlustfunktionen'
# compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare loss functions MSE, MAE
export_name = 'Comp_Loss_MSE_MAE_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-16', 'Gievenbeck_LSTM_Single_MAE2024-05-16']
model_alias = ['"Loss MSE"','"Loss MAE"']
base_folder = os.path.join('05_models', 'loss_functions_compare')
title = 'Vergleich Verlustfunktionen'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

###################################################
# Compare shuffle = True and shuffle = False
export_name = 'Comp_Shuffle_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-16', 'Gievenbeck_LSTM_Single_MSE_No_Shuffle_2024-05-15']
model_alias = ['"Gemischt"','"Geordnet"']
base_folder = os.path.join('05_models', 'shuffle_compare')
title = 'Vergleich Shuffle'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare Units = 32, Units = 64 and Units = 128
export_name = 'Comp_Units_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE2024-05-16', 'Gievenbeck_LSTM_Single_MSE_u64_2024-05-16', 'Gievenbeck_LSTM_Single_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_Single_MSE_u256_2024-05-16']
model_alias = ['"32"','"64"','"128"', '"256"']
base_folder = os.path.join('05_models', 'units_compare')
title = 'Vergleich Neuronenanzahl'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare Shallow and Deep
export_name = 'Comp_Deep_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_Double_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16']
model_alias = ['"1x128"','"2x128"','"3x128"']
base_folder = os.path.join('05_models', 'deep_compare')
title = 'Vergleich Schichtanzahl'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare Deep 5x
export_name = 'Comp_Deep_5x_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_5Layers_MSE_u128_2024-05-17']
model_alias = ['"Dreifach"','"Fünffach"']
base_folder = os.path.join('05_models', 'deep_compare')
title = 'Vergleich Schichtanzahl 5x'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# # Compare Shallow and Deep with different units
# export_name = 'Comp_Deep_' + str(date.today()) + '.ipynb'
# model_names = ['Gievenbeck_LSTM_Single_MSE_u128_2024-05-14', 'Gievenbeck_LSTM_Double_MSE_u64_2024-05-15', 'Gievenbeck_LSTM_Triple_MSE_u32_2024-05-15']
# model_alias = ['"1 Schicht"','"2 Schichten"','"3 Schichten"']
# base_folder = os.path.join('05_models', 'deep_compare')
# compare_models(model_names, model_alias, export_name, base_folder)

# Compare accum precip
export_name = 'Comp_accum_precip_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_Triple_MSE_u128_accum_2024-05-17']
model_alias = ['"Normal"','"Akkumuliert"']
base_folder = os.path.join('05_models', 'accum_compare')
title = 'Mit und ohne bekanntem Abfluss'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare q Known and q Unknown
export_name = 'Comp_qKnown_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16', 'Gievenbeck_LSTM_Triple_MSE_u128_qKnown_2024-05-18']
model_alias = ['"Q unbekannt"','"Q bekannt"']
base_folder = os.path.join('05_models', 'q_known_compare')
title = 'Mit und ohne bekanntem Abfluss'
compare_models(model_names, model_alias, export_name, base_folder, title = title)


# Compare Final Model
export_name = 'Comp_final_' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MAE2024-05-16', 'Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16']
model_alias = ['"Initial"','"Final"']
base_folder = os.path.join('05_models', 'final_compare')
title = 'Vergleich des Finalen Modells'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

# Compare Final Model with GPU
export_name = 'Comp_final_GPU' + str(date.today()) + '.ipynb'
model_names = ['Gievenbeck_LSTM_Single_MAE_GPU2024-05-19', 'Gievenbeck_LSTM_Triple_MSE_u128_GPU_2024-05-19']
model_alias = ['"Initial GPU"','"Final GPU"']
base_folder = os.path.join('05_models', 'final_compare')
title = 'Vergleich des Finalen Modells mit Grafikkarte'
compare_models(model_names, model_alias, export_name, base_folder, title = title)

#### Compare new features





# # Test RR
# export_name = 'Test_RR' + '_' + str(date.today()) + '.ipynb'
# model_names = ['Gievenbeck_RR_20240507']
# model_alias = ['"RR"']
# base_folder = os.path.join('05_models')
# compare_models(model_names, model_alias, export_name, base_folder)
