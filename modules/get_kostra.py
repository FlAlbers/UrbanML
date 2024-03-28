


import os
import pandas as pd
import sys
from modules.py_kostra_master import pykostra as pyk
from modules.py_kostra_master import download

def get_kostra_by_index_rc(folder_path, index_rc, download=False):
    """
    This function retrieves data from multiple CSV files in a specified folder,
    filters the data based on the INDEX_RC column, performs some data manipulation,
    and saves the filtered data to a new CSV file.

    Parameters:
    - folder_path (str): The path to the folder containing the CSV files.
    - index_rc (int): The value to filter the data on the INDEX_RC column.
    - download (bool): Whether to download raw data before processing. Default is False.

    """

    raw_dir = os.path.join(folder_path, 'raw')
    unzip_dir = os.path.join(folder_path, 'unzip')

    if download:
        download.get_raw(type='tab', raw_dir=raw_dir)
        pyk.get_csv_files(raw_dir=raw_dir, unzip_dir=unzip_dir)

    file_list = [file_name for file_name in os.listdir(unzip_dir) if file_name.startswith('StatRR')]

    combined_df = pd.DataFrame()

    for file_name in file_list:
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(unzip_dir, file_name), delimiter=';', decimal=',')
            df = pd.DataFrame(df)

            df_filtered = df[df['INDEX_RC'] == index_rc]

            table_name = file_name[:-4][-10:]

            df_filtered['duration'] = file_name[:-4][-5:].lstrip('0')
            df_filtered.set_index('duration', inplace=True)

            if combined_df.empty == True:
                combined_df = df_filtered
            else:
                combined_df = pd.concat([combined_df, df_filtered])

    combined_df = combined_df.replace(',', '.', regex=True)
    combined_df = combined_df.astype(float, errors='ignore')

    selected_columns = ['HN_001A', 'HN_002A', 'HN_003A', 'HN_005A', 'HN_010A', 'HN_020A', 'HN_030A', 'HN_050A', 'HN_100A']
    combined_df = combined_df[selected_columns].round(2)

    combined_df.columns = ['1', '2', '3', '5', '10', '20', '30', '50', '100']

    savePath = os.path.join(folder_path, f'kostra_{index_rc}.csv')
    combined_df.to_csv(savePath, sep=',', header=True)


if __name__ == '__main__':
    folder_path = os.path.join(os.getcwd(), '02_input_data')
    get_kostra_by_index_rc(folder_path, 118111)


