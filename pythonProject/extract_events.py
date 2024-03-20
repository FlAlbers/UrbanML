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

    # Order data in P_events by hN_mm
    P_events_sort = P_events.sort_values('hN_mm')

    # reset index of P_events_sort
    P_events_sort = P_events_sort.reset_index(drop=True)

    n_events = [5,15,20,25,35]
    distribution = [0, 0.40, 0.70, 0.90, 0.975, 1]

    P_sample = pd.DataFrame()
    for i in range(len(n_events)):
        # Select data from P_events based on index range
        selected_events = P_events_sort[(P_events.index >= len(P_events) * distribution[i]) & (P_events.index < len(P_events) * distribution[i+1])]

        # Select a sample from selected_events
        P_sample = pd.concat([P_sample, selected_events.sample(n_events[i], random_state=2).sort_values('hN_mm')])

    print(P_sample)

    P_sample = P_sample.reset_index(drop=True)
    
    # Plot index vs hN_mm
    plt.plot(P_sample.index, P_sample['hN_mm'])
    plt.xlabel('Index')
    plt.ylabel('hN_mm')
    plt.title('Index vs hN_mm')
    plt.show()

    # Plot index vs hN_mm
    plt.plot(P_events_sort.index, P_events_sort['hN_mm'])
    plt.xlabel('Index')
    plt.ylabel('hN_mm')
    plt.title('Index vs hN_mm')
    plt.show()

    extract_events(events_path, P_path, save_folder)


#####################
    n_events = [33,33,34]
    distribution = [0, 0.05, 0.10, 1]

    P_sample = pd.DataFrame()
    for i in range(len(n_events)):
        # Select data from P_events based on index range
        selected_events = P_events_sort[(P_events["hN_mm"] >= P_events["hN_mm"].max() * distribution[i]) & (P_events["hN_mm"] < P_events["hN_mm"].max() * distribution[i+1])]

        # Select a sample from selected_events
        P_sample = pd.concat([P_sample, selected_events.sample(n_events[i], random_state=1).sort_values('hN_mm')])

    print(len(P_sample))

    # P_sample = P_sample.reset_index(drop=True)


    # Plot index vs hN_mm
    plt.plot(P_sample.index, P_sample['hN_mm'])
    plt.xlabel('Index')
    plt.ylabel('hN_mm')
    plt.title('Index vs hN_mm')
    plt.show()




