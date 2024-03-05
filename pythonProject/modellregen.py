
import sys
sys.path.insert(0, r'C:\Users\Laptop-F-Albers\PycharmProjects\urbanml\pythonProject')

import matplotlib.pyplot as plt
import os
import pandas as pd
import kostra
from datetime import timedelta

folder_path = r'C:\Users\Laptop-F-Albers\PycharmProjects\urbanml\pythonProject\Kostra'
time_kostra = 720


#kostra_118111 = kostra.get_kostra_by_index_rc(folder_path,118111, 'urbanml\\pythonProject\\')
kostra_118111 = pd.read_csv('C:\\Users\\Laptop-F-Albers\\PycharmProjects\\urbanml\\pythonProject\\kostra_118111.csv', delimiter=';')
kostra_118111['dauer'] = pd.to_timedelta(kostra_118111['dauer'].astype(float), unit='m')
kostra_118111.set_index('dauer', inplace=True)
# Create a new DataFrame with index '0 days 00:00:00' and all values 0
new_row = pd.DataFrame(0, index=[pd.to_timedelta('0 days 00:00:00')], columns=kostra_118111.columns)
# Concatenate the new row with kostra_118111
kostra_118111 = pd.concat([new_row, kostra_118111])
# Sort the DataFrame by index
# kostra_118111 = kostra_118111.sort_index()
delta_h = kostra_118111
delta_h = delta_h.rename_axis('dauer')

# kostra_118111['HN_001A'].plot(x='deltatime')
# plt.show()
# delta_h['HN_001A'].plot()
# plt.show()


# Create a time series with 5-minute intervals
# time_series = pd.date_range(start='1/1/2018', freq='5min', periods= 576).time # 288
time_series = range(0, time_kostra + 5, 5)
time_series = pd.to_timedelta(time_series, unit='m')
# Convert time objects to timedelta
#time_series = [timedelta(hours=t.hour, minutes=t.minute, seconds=t.second) for t in time_series]

# Create an empty DataFrame using this time series
kostra_d = pd.DataFrame(index=time_series)
kostra_d = kostra_d.reset_index()
kostra_d = kostra_d.rename(columns={'index': 'dauer'})
kostra_d['x'] = 0

kostra_d = pd.merge(kostra_d, delta_h, on='dauer', how='left')
kostra_d.set_index('dauer', inplace=True)

kostra_d = kostra_d.interpolate('linear')
kostra_d = kostra_d.fillna(0)

kostra_d = kostra_d.diff()

eulerI = kostra_d

#Euler 1
max_dauer = eulerI.index.max()
selected_kostra = eulerI[(eulerI.index <= 0.3 * max_dauer) & (eulerI.index > 0 * max_dauer)]


selected_kostra["HN_001A"] = selected_kostra["HN_001A"].values[::-1]
eulerI = pd.concat([eulerI[(eulerI.index > 0.3 * max_dauer)], selected_kostra])
eulerI = eulerI.sort_index()

eulerI['HN_001A'].plot()
plt.show()


kostra_d['HN_001A'].plot()
plt.show()






