
import os
import pandas as pd

# folder_path = r'C:\Users\Laptop-F-Albers\PycharmProjects\urbanml\pythonProject\Kostra'

def get_kostra_by_index_rc(folder_path, index_rc, savePath=''):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    combined_df = pd.DataFrame()
    # Iterate over each file
    for file_name in file_list:
        if file_name.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(folder_path, file_name), delimiter=';', decimal=',')
            df = pd.DataFrame(df)

            # Filter the DataFrame for INDEX_RC
            df_filtered = df[df['INDEX_RC'] == index_rc]
            
            # Get the last 6 characters of the file name (excluding the extension)
            table_name = file_name[:-4][-10:]
            
            # Add dauer to df_filtered
            df_filtered['dauer'] = file_name[:-4][-5:]
            df_filtered.set_index('dauer', inplace=True)

            # Append the filtered DataFrame to a new DataFrame
            if combined_df.empty == True:
                combined_df = df_filtered
            else:
                combined_df = pd.concat([combined_df, df_filtered])

     
    combined_df = combined_df.replace(',', '.', regex=True)
    combined_df = combined_df.astype(float, errors='ignore')
    combined_df.to_csv(f'{savePath}kostra_{index_rc}.csv', sep=';',header=True)

        

        
