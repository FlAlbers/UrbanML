from concurrent.futures import ThreadPoolExecutor
from pyswmm import Simulation
import os
import time
from datetime import datetime
from modules.save_load_model import load_model_container
from modules.sequence_and_normalize import sequence_list
from tensorflow.keras.models import model_from_json
from tensorflow.keras import utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd


##### Calculate elapsed time for SWMM simulation
sim_path = '03_sim_data\\Gievenbeck_e2_T2D30_Time_compare.inp'

with Simulation(sim_path) as sim:
    print("\nSimulation info:")
    print(f"Title: {sim_path}")
    print("Start Time: {}".format(sim.start_time))
    start_time = datetime.now()
    for step in sim:
        pass
    end_time = datetime.now()
    elapsed_time_swmm = end_time - start_time
    hours, remainder = divmod(elapsed_time_swmm.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = elapsed_time_swmm.microseconds // 1000
    print("Start Time: ", start_time.strftime("%H:%M:%S:%f")[:-3])
    print("End Time: ", end_time.strftime("%H:%M:%S:%f")[:-3])
    print("Elapsed Time: ", "{:02}:{:02}:{:02}:{:03}".format(hours, minutes, seconds, milliseconds))



########### Calculate elapsed time for LSTM model 

model_name = 'Gievenbeck_LSTM_Triple_MSE_u128_2024-05-16'
base_folder = os.path.join('05_models', 'final_compare')
model_folder = os.path.join(base_folder, model_name)
# model_container = load_model_container(model_folder, print_info=False)
models = []
model_folder = os.path.join(base_folder, model_name)
model_container = load_model_container(model_folder, print_info=False)
model_id = 'model_' + str(model_container['select_id'])
# model = model_container[model_id]
model = model_container['selected_model']
model['cv_scores'] = model_container['cv_scores']
comb_history = {}
for key in model_container['selected_model']['history'].keys():
    sel_history = model_container['selected_model']['history'][key]
    prev_history = model_container[model_id]['history'][key]
    comb_history[key] = prev_history + sel_history
model['combined_history'] = comb_history

models.append(model)

m=models[0]
# for m in models:
model = m['model']
lag = m['lag']
delay = m['delay']
p_steps = m['prediction_steps']
in_vars_future = m['in_vars']
try:
    in_vars_past = m['in_vars_past']
except:
    in_vars_past = None
    pass
out_vars = m['out_vars']
test_data = m['test_data']
train_data = m['train_data']
# val_data = m['validation_data']
in_scaler = m['in_scaler']
out_scaler = m['out_scaler']

seq_test, seq_test_trans = sequence_list(test_data, in_vars_future=in_vars_future, out_vars=out_vars, in_scaler=in_scaler, 
                                out_scaler=out_scaler, lag=lag, delay=delay, prediction_steps=p_steps, in_vars_past=in_vars_past)


m.update({'seq_test':seq_test, 'seq_test_trans':seq_test_trans})


seq_test = m['seq_test']
seq_test_trans = m['seq_test_trans']
model = m['model']

test_time_seq = seq_test_trans[0][1]
print(len(seq_test_trans))
for i in range(len(seq_test_trans)-1):
    
    test_time_seq = np.concatenate((test_time_seq, seq_test_trans[i + 1][1]), axis=0)
    # test_time_seq = test_time_seq + seq_test_trans[i][1]

len(test_time_seq)


# test_time_seq = seq_test_trans[3][1][12]
# test_time_seq = [np.expand_dims(test_time_seq, axis=0)]

from datetime import datetime
# get start time
start_time = datetime.now()
# Predict the sequence
Predict = model.predict(test_time_seq, verbose=0)
Predict_invert = out_scaler.inverse_transform(Predict)
# Get end time
end_time = datetime.now()
# print(Predict_invert)
elapsed_time_LSTM_all = end_time - start_time
hours, remainder = divmod(elapsed_time_LSTM_all.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
milliseconds = elapsed_time_LSTM_all.microseconds // 1000
print("Model: ", m['name'])
print("Beginn: ", start_time.strftime("%H:%M:%S:%f")[:-3])
print("Ende: ", end_time.strftime("%H:%M:%S:%f")[:-3])
print("Rechenzeit aller Testsequenzen: ", "{:2}.{:03}".format(seconds,milliseconds), "s")
print("\n")

print(len(Predict_invert))





##### Calculate elapsed time for LSTM model single sequence

test_time_seq = seq_test_trans[3][1][12]
test_time_seq = [np.expand_dims(test_time_seq, axis=0)]

from datetime import datetime
# get start time
start_time = datetime.now()
# Predict the sequence
Predict = model.predict(test_time_seq, verbose=0)
Predict_invert = out_scaler.inverse_transform(Predict)
# Get end time
end_time = datetime.now()
# print(Predict_invert)
elapsed_time_LSTM_single = end_time - start_time
hours, remainder = divmod(elapsed_time_LSTM_single.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
milliseconds = elapsed_time_LSTM_single.microseconds // 1000
print("Model: ", m['name'])
print("Beginn: ", start_time.strftime("%H:%M:%S:%f")[:-3])
print("Ende: ", end_time.strftime("%H:%M:%S:%f")[:-3])
print("Rechenzeit einer Sequenz: ", "{:2}.{:03}".format(seconds,milliseconds), "s")
print("\n")

print(len(Predict_invert))


ms_LSTM_single = elapsed_time_LSTM_single.total_seconds() * 1000
ms_LSTM_all = elapsed_time_LSTM_all.total_seconds() * 1000
ms_SWMM = elapsed_time_swmm.total_seconds() * 1000

print("Elapsed time LSTM single sequence: ", round(ms_LSTM_single,0), "ms")
print("Elapsed time LSTM all sequences: ", round(ms_LSTM_all,0), "ms")
print("Elapsed time SWMM: ", round(ms_SWMM,0), "ms")


# Train times
# import random
times = pd.DataFrame(columns=['Model', 'Calc Time'])
new_row = pd.DataFrame([{'Model': 'LSTM 1788x', 'Calc Time': round(ms_LSTM_all,0) }])
times = pd.concat([times,new_row], ignore_index=True)
new_row = pd.DataFrame([{'Model': 'LSTM 1x', 'Calc Time': round(ms_LSTM_single,0) }])
times = pd.concat([times,new_row], ignore_index=True)
new_row = pd.DataFrame([{'Model': 'SWMM', 'Calc Time': round(ms_SWMM,0) }])
times = pd.concat([times,new_row], ignore_index=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 2))
# plt.barh(times['Model'], times['Calc Time'])
bars = plt.barh(times['Model'], times['Calc Time'])

# Add labels and title
plt.xlabel('Zeit - T [ms]')
plt.ylabel('Modell')
plt.title('Berechnungszeiten in Millisekunden')

# Add actual numbers to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:.0f}',
             ha='left', va='center', rotation=-90)

# Display the plot
plt.tight_layout()
plt.show()

