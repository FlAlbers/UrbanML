import matplotlib.pyplot as plt
import numpy as np


def plot_seq_i_d_Q(in_seq, out_act, out_pred, event_meta, interval):
    '''
    Single plot for one sequence with duration, intensity and discharge
        - in_seq: input sequence with duration and intensity
        - out_act: actual output sequence with discharge and duration
        - out_pred: predicted output sequence with discharge and duration

    Returns:
        - plot with two y-axes for precipitation intensity and discharge and a x-axis with duration
    '''
    fig, axs = plt.subplots(figsize=(6, 4))

    ax1 = axs
    ax1.set_title(f"Ereignis: {event_meta['Ereignis']}, {round(event_meta['total precipitation'])} mm, {event_meta['duration']} min", pad=20)
    x = in_seq[:,0]  # Set x-axis values
    ax1.bar(x, in_seq[:,1], color='blue', label='iN', width=interval, align='edge')
    top_lim = max(max(in_seq[:,1]), 50)
    ax1.set_ylim(bottom=0, top=top_lim)  # Set y-axis to start from zero
    ax1.set_ylabel('Niederschlagsintensität iN [mm/h]')
    ax1.set_xlabel('Ereignisdauer [min]')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, fontsize='small')

    # Create a twin axis on the right side
    # Plotting the predicted and actual values in the corresponding subplot
    ax2 = ax1.twinx()
    ax2.plot(out_pred[:, 0], out_pred[:, 1], color='red', label='Q Vorh.')
    ax2.plot(out_act[:, 0], out_act[:, 1], color='green', label='Q Sim.')
    ax2.set_ylim(bottom=0)  # Set y-axis to start from zero
    ax2.set_ylabel('Abfluss Q [m³/s]')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.3, 1.1), frameon=False, ncol=2, fontsize='small')

    return plt



# Function testing Area
if __name__ == '__main__':
    in_seq = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    out_pred = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    out_act = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    event_meta = {'Ereignis': 'Aufgezeichnet', 'duration': 10, 'total precipitation': 100, 'max intensity': 10, 'interval': 1}
    interval = 1
    plot_seq_i_d_Q(in_seq, out_pred, out_act, event_meta, interval).show()