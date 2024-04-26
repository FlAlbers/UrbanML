import papermill as pm
import os
from datetime import date
import subprocess



base_name = 'model_testing.ipynb'
new_name = 'Comp_Threshold_' + str(date.today()) + '.ipynb'
base_path = os.path.join(os.getcwd(), base_name)
new_path = os.path.join(os.getcwd(), '07_model_compare',  new_name)
output_format = 'html'

model_names = ['Gievenbeck_LSTM_Single_Thresh_1h_P_20240408','Gievenbeck_LSTM_Single_CV_1h_P_20240408']
model_alias = ['"Threshold"','"No Threshold"']

res = pm.execute_notebook(
    base_path,
    new_path,
    parameters = dict(model_names=model_names, model_alias=model_alias)
)

# Export the notebook to HTML format
subprocess.run(['jupyter', 'nbconvert', '--to', output_format, '--no-input', new_path])