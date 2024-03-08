
import sys
import os
current_path = os.getcwd()
sys.path.insert(0, os.path.join(current_path, 'pythonProject')) #working directory needs to be \urbanml


import matplotlib.pyplot as plt
import pandas as pd
import kostra
from datetime import timedelta

# path to the kostra raw data
Kostra_raw_path = os.path.join(current_path, 'pythonProject\\Kostra')
# folder_path = r'C:\Users\Laptop-F-Albers\PycharmProjects\urbanml\pythonProject\Kostra' #old path


## Get the kostra data for INDEX_RC 118111
#kostra_118111 = kostra.get_kostra_by_index_rc(Kostra_raw_path,118111, 'urbanml\\pythonProject\\')

# current path to the kostra data extracted by kostra.py
kostra_data_path = os.path.join(current_path, 'pythonProject\\kostra_118111.csv')


kostra_118111 = pd.read_csv(kostra_data_path, delimiter=',')
kostra_118111['duration'] = pd.to_timedelta(kostra_118111['duration'].astype(float), unit='m')
kostra_118111.set_index('duration', inplace=True)
# selected_columns = ['HN_001A', 'HN_002A', 'HN_003A', 'HN_005A', 'HN_010A', 'HN_020A', 'HN_030A', 'HN_050A', 'HN_100A']
# kostra_118111 = kostra_118111[selected_columns]
# kostra_118111.columns = ['1', '2', '3', '5', '10', '20', '30', '50', '100']


jaerlichkeiten = kostra_118111.columns
dauern = kostra_118111.index.total_seconds() / 60

# Create a new DataFrame with index '0 days 00:00:00' and all values 0
new_row = pd.DataFrame(0, index=[pd.to_timedelta('0 days 00:00:00')], columns=kostra_118111.columns)
# Concatenate the new row with kostra_118111
kostra_118111 = pd.concat([new_row, kostra_118111])

delta_h = kostra_118111
delta_h = delta_h.rename_axis('duration')



for j in jaerlichkeiten:
    for d in dauern:
        # Create a time series with 5-minute intervals
        # time_series = pd.date_range(start='1/1/2018', freq='5min', periods= 576).time # 288
        time_series = range(0, int(d) + 5, 5)
        time_series = pd.to_timedelta(time_series, unit='m')

        # Create an empty DataFrame using this time series
        kostra_d = pd.DataFrame(index=time_series)
        kostra_d = kostra_d.reset_index()
        kostra_d = kostra_d.rename(columns={'index': 'duration'})
        kostra_d['x'] = 0

        kostra_d = pd.merge(kostra_d, delta_h[j], on='duration', how='left')
        kostra_d.set_index('duration', inplace=True)

        kostra_d = kostra_d.interpolate('linear')
        kostra_d = kostra_d.fillna(0)
        kostra_d = kostra_d.diff()

        euler2 = kostra_d

        #Euler 2: select precip data and reverse first 30% of the data
        max_dauer = euler2.index.max()
        selected_kostra = euler2[(euler2.index <= (1/3) * max_dauer) & (euler2.index > 0 * max_dauer)]

        selected_kostra[j] = selected_kostra[j].values[::-1]
        euler2 = pd.concat([euler2[(euler2.index > (1/3) * max_dauer)], selected_kostra])
        euler2 = euler2.sort_index()
        index_min = euler2.index.total_seconds() / 60
        euler2.index = index_min.astype(int)
        euler2[j].to_csv(f'pythonProject\\climate_data\\euler2_{j}a_{int(d)}.csv', sep=',', header=True, index=True)


# kostra_d['HN_001A'].plot()
# plt.show()



