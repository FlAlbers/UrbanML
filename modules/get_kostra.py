
import os
import pandas as pd
import sys
# print(os.getcwd())
from modules.py_kostra_master import pykostra as pyk
from modules.py_kostra_master import download

# folder_path = '02_input_data\\Kostra'

def get_kostra_by_index_rc(folder_path, index_rc, download=False):
    raw_dir=os.path.join(folder_path, 'raw')
    unzip_dir=os.path.join(folder_path, 'unzip')
    if download:
        download.get_raw(type='tab', raw_dir=raw_dir)
        pyk.get_csv_files(raw_dir=raw_dir,unzip_dir=unzip_dir)
    # Get a list of all files in the folder
    file_list = [file_name for file_name in os.listdir(unzip_dir) if file_name.startswith('StatRR')]

    combined_df = pd.DataFrame()
    # Iterate over each file
    for file_name in file_list:
        if file_name.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(unzip_dir, file_name), delimiter=';', decimal=',')
            df = pd.DataFrame(df)

            
            # Filter the DataFrame for INDEX_RC
            df_filtered = df[df['INDEX_RC'] == index_rc]
            
            print(df_filtered)

            # Get the last 6 characters of the file name (excluding the extension)
            table_name = file_name[:-4][-10:]
            
            # Add duration to df_filtered
            df_filtered['duration'] = file_name[:-4][-5:].lstrip('0')
            df_filtered.set_index('duration', inplace=True)

            # Append the filtered DataFrame to a new DataFrame
            if combined_df.empty == True:
                combined_df = df_filtered
            else:
                combined_df = pd.concat([combined_df, df_filtered])

    # Replace commas with dots and convert to float
    combined_df = combined_df.replace(',', '.', regex=True)
    combined_df = combined_df.astype(float, errors='ignore')

    #rename columns and round data to 2 decimal places
    selected_columns = ['HN_001A', 'HN_002A', 'HN_003A', 'HN_005A', 'HN_010A', 'HN_020A', 'HN_030A', 'HN_050A', 'HN_100A']
    combined_df = combined_df[selected_columns].round(2)

    #rename columns
    combined_df.columns = ['1', '2', '3', '5', '10', '20', '30', '50', '100']

    # Save the DataFrame to a CSV file
    savePath = os.path.join(folder_path, f'kostra_{index_rc}.csv')
    combined_df.to_csv(savePath, sep=',',header=True)


if __name__ == '__main__':
    # get_kostra_by_index_rc(folder_path, 118111, 'pythonProject\\')
    folder_path = os.path.join(os.getcwd(),'02_input_data')
    get_kostra_by_index_rc(folder_path,118111)


