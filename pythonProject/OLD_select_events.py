import pandas as pd
import matplotlib.pyplot as plt



def extract_events(P_events, P_series, save_folder):
    # Convert date in P_FMO to datetime
    first_col = P_series.columns[0]
    P_series[first_col] = pd.to_datetime(P_series[first_col])
    P_series[first_col] = P_series[first_col].dt.tz_localize(None)

    # Convert start and end in events_FMO to datetime
    P_events['start'] = pd.to_datetime(P_events['start'])
    P_events['end'] = pd.to_datetime(P_events['end'])

    # test start and end
    start = P_events['start']
    end = P_events['end']
    hN = P_events['hN_mm']

    # Iterate through all events
    for i in range(len(start)):
        # Select timeframe in P_series of start and end
        event_series = P_series[(P_series['date'] >= start[i]) & (P_series['date'] <= end[i])]

        # Write the extracted event to a CSV file
        start_name = str(start[i]).replace(':', ' ')
        save_path = f'{save_folder}\\{start_name}_hN{hN[i]}.csv'
        event_series.to_csv(save_path, index=False, header=True)
        print(f'Event {start[i]} saved to {save_folder}\\{start_name}_hN{hN[i]}.csv')


if __name__ == '__main__':
    events_path = 'pythonProject\\events_FMO.csv'
    P_path = 'pythonProject\\P_FMO.csv'
    save_folder = 'pythonProject\\events_FMO'

    P_events = pd.read_csv(events_path)
    P_series = pd.read_csv(P_path)

    P_events_sample = P_events.sample(10, random_state=1)

    # test extract_events with 10 random samples
    extract_events(events_path, P_path, save_folder)

