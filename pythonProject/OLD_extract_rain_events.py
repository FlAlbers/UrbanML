import numpy as np
import pandas as pd

klima_1766 = pd.read_csv("pythonProject\\klima_1766.csv")
klima_1766 = klima_1766[['date', 'precipitation_height']]
klima_1766.set_index('date', inplace=True)
klima_1766 = klima_1766[:1000]
klima_1766['precipitation_height'].sum()

def run_sum_max(x, n):
    sz = len(x)
    res = np.zeros(sz)
    for i in range(sz - n + 1):
        res[i + n - 1] = np.sum(x.iloc[i:i+n])
    return np.max(res)

def rainfall_events(x, value_threshold=0.01, event_threshold=0.5, n_max_event_time=48, n_roll_max_event=12):
    n = len(x)
    m = 0
    event = False
    event_mat = np.full((n, 6), np.nan)
    event_idx = 0

    # print(x.iloc[0])

    for i in range(n):
        if event == False and x['precipitation_height'].iloc[i] <= value_threshold:
            pass
        elif event == True and x['precipitation_height'].iloc[i] > value_threshold:
            pass
        else:
            m = min(n - 1, i + n_max_event_time)
            sum_event_follow = np.sum(x['precipitation_height'].iloc[i:m+1])

            if not event and sum_event_follow > event_threshold:
                event = True
                event_mat[event_idx, 0] = i
            elif event and sum_event_follow <= event_threshold:
                event = False
                event_mat[event_idx, 1] = i
                event_idx += 1

    event_mat = event_mat[:max(event_idx, 1), :] ### unklar was das macht

    for i in range(event_idx):
        event_vals = x[int(event_mat[i, 0]):int(event_mat[i, 1])+1]
        event_mat[i, 2] = np.sum(event_vals)
        event_mat[i, 3] = np.mean(event_vals)
        event_mat[i, 4] = np.max(event_vals)
        event_mat[i, 5] = run_sum_max(event_vals, n_roll_max_event)

    return event_mat


# klima_1766.iloc[1]
events = rainfall_events(klima_1766[['precipitation_height']])
events