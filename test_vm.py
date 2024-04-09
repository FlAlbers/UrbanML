
import numpy as np
import pandas as pd
from modules.extract_sim_data import single_node
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input
from keras.layers import LSTM
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from modules.sequence_and_normalize import sequence_data, sequence_sample_random, sequence_list
import os
import joblib
import pickle



folder_path_sim = os.path.join('03_sim_data', 'inp')
sims_data = single_node(folder_path_sim, 'R0019769',resample = '5min')