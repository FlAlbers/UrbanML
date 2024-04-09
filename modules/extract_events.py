
"""
    Author: Flemming Albers
    Extracts rainevents from a given timeseries based on start and end dates.
    Extracted events are saved as CSV files in a specified folder.

    Args:
        P_events (DataFrame): DataFrame containing event information with 'start'and 'end' of the event.
        P_series (DataFrame): DataFrame containing time series of precipitation data.
        save_folder (str): Path to the folder where the extracted event files will be saved.

    """

import pandas as pd
import os
import matplotlib.pyplot as plt

def extract_events(P_events, P_series, save_folder):
    
    # Convert date in P_series to datetime
    first_col = P_series.columns[0]
    P_series[first_col] = pd.to_datetime(P_series[first_col])
    P_series[first_col] = P_series[first_col].dt.tz_localize(None)

    # Convert start and end in P_events to datetime
    P_events['start'] = pd.to_datetime(P_events['start'])
    P_events['end'] = pd.to_datetime(P_events['end'])

    # Extract events
    start = pd.to_datetime(P_events['start'].values.flatten())
    end = pd.to_datetime(P_events['end'].values.flatten())
    hN = P_events['hN_mm'].values.flatten()

    # Iterate through all events
    for i in range(len(start)):
        # Select timeframe in P_series of start and end
        event_series = P_series[(P_series['date'] >= start[i]) & (P_series['date'] <= end[i])]

        # Write the extracted event to a CSV file
        start_name = str(start[i]).replace(':', ' ')

        # Create folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = f'{save_folder}\\{start_name}_hN{hN[i]}.csv'
        event_series.to_csv(save_path, index=False, header=True)
        print(f'Event {start[i]} saved to {save_folder}\\{start_name}_hN{hN[i]}.csv')



#Code for Testing function
if __name__ == '__main__':
    events_path = os.path.join('02_input_data','events_FMO.csv')
    P_path = os.path.join('02_input_data','P_FMO.csv')
    save_folder = os.path.join('03_sim_data','quick_sim')
    # events_path = '02_input_data\\events_FMO.csv'
    # P_path = '02_input_data\\P_FMO.csv'
    # save_folder = '03_sim_data\\quick_sim'

    P_events = pd.read_csv(events_path)
    P_series = pd.read_csv(P_path)

    P_events_sample = P_events.sample(10, random_state=1)

    # Test extract_events with 10 random samples
    extract_events(P_events_sample, P_series, save_folder)

