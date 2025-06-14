# input regex for where to find the date info
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib.animation import FuncAnimation
from scipy import signal


epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)

# input: 2d array
# output: list of 2d array
# window_length, stride are in second
def make_ani_data(data, dx, dt, window_length, stride):
    data_list = list()
    nx, nt = data.shape
    x = int (nx * dx )
    t = int (nt * dt)
    print (t)
    data_list_len = int( (t - window_length)//stride + 1 )
    print (data_list_len)
    for i in range(data_list_len):
        start_idx = int( (0+stride*i)*1//dt )
        end_idx = int( (window_length+stride*i)*1//dt )
        data_temp = data[:,start_idx:end_idx]
        data_list.append(data_temp)
    return data_list

def animate_heatmap(data, channels, dx, dt, interval=1000, st=None, stride=1):
    """
    Parameters:
    - data: 3D numpy array or list of 2D arrays, where each 2D array represents the heatmap data at a time step.
    - interval: Delay between frames in milliseconds.
    """
    nx, nt = data[0].shape
    x = channels * dx
    t = np.arange(nt) * dt
    title = st.strftime('%H:%M:%S, %m/%d, %Y')
    # Set up the figure and axis
    fig, ax = plt.subplots()
    
    # Initialize the heatmap with the first frame
    heatmap = ax.imshow(normalize(data[0]).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", 
               extent=[x[0], x[-1], t[-1], t[0]], interpolation="none", animated=True)
    ax.set_xlabel("Channel Number")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    # Function to update the heatmap at each frame
    def update(frame):
        stt = st + timedelta(seconds=stride*frame)
        title = stt.strftime('%H:%M:%S, %m/%d, %Y')
        heatmap.set_data(normalize(data[frame]).T)
        ax.set_title(f"{title} (Frame {frame+1}/{len(data)})")
        return heatmap,
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(data), interval=interval, blit=True)
    
    return anim

def plot_das_data(data, channels, dx, dt, start_time=None, end_time=None, title=None, data_source=None, deck_splits=None, deck_names=None,
                 velocity_line=None):
    nx, nt = data.shape
    x = channels * dx
    t = np.arange(nt) * dt

    if start_time is not None and end_time is not None:
        time_indices = (t >= start_time) & (t <= end_time)
        data = data[:, time_indices]
        t = t[time_indices]
    #plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", 
               extent=[x[0], x[-1], t[-1], t[0]], interpolation="none", animated=True)
    ax.set_xlabel("Channel Position (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)


    # --- Optional: Plot velocity line ---
    if velocity_line is not None and velocity_line != 0:
        (x1, t1), (x2, t2) = velocity_line
        ax.plot([x1, x2], [t1, t2], color='yellow', linewidth=2, linestyle='--', label='Velocity Line')
        
    
    # Add deck splits if data source is "Newville"
    if data_source == "Newville" and deck_splits is not None:
        split_positions = [channels[split] * dx for split in deck_splits if 0 <= split < len(channels)]
        for pos in split_positions:
            ax.axvline(pos, color='red', linestyle='--', linewidth=1.5)
            
        # Add deck labels
        if deck_names is not None and len(deck_names) == len(split_positions) + 1:
            label_edges = [x[0]] + split_positions + [x[-1]]
            for i in range(len(deck_names)):
                x_center = (label_edges[i] + label_edges[i + 1]) / 2
                y_center = (t[0] + t[-1]) / 2
                ax.text(
                    x_center, y_center, deck_names[i],
                    color='blue', fontsize=12, fontweight='bold',
                    ha='center', va='center', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )

        if len(split_positions) % 2 == 1:
            mid_idx = len(split_positions) // 2
            mid_pos = split_positions[mid_idx]

            # Middle vertical split line and label
            ax.axvline(mid_pos, color='red', linestyle='-', linewidth=1.5)

            # Horizontal line just above x-axis (bottom of image)
            y_line = t[0] - 0.01 * (t[0] - t[-1])
            x_left = [x[0], mid_pos]
            x_right = [mid_pos, x[-1]]

            # Plot horizontal lines along bottom edge
            ax.plot(x_left, [y_line, y_line], color='green', linewidth=3, label="Coupled")
            ax.plot(x_right, [y_line, y_line], color='yellow', linewidth=3, label="Moderately uncoupled")

            # Move legend above the figure
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=True)
            
    fig.tight_layout()
    fig.show()



def plot_single(data, channel_num, dx, dt, start_time=None, end_time=None):
    """
    Plot a single DAS channel's time series over a specified time range.
    
    Parameters:
    - data: 2D ndarray of shape (n_channels, n_times)
    - channel_num: index of the channel to plot
    - dx: spatial sampling (m)
    - dt: time sampling (s)
    - start_time, end_time: time window to plot (in seconds)
    """
    if channel_num < 0 or channel_num >= data.shape[0]:
        raise ValueError(f"channel_num must be between 0 and {data.shape[0]-1}")
    
    nt = data.shape[1]
    t = np.arange(nt) * dt
    signal = data[channel_num, :]
    
    # Apply time range
    if start_time is not None and end_time is not None:
        time_mask = (t >= start_time) & (t <= end_time)
        t = t[time_mask]
        signal = signal[time_mask]
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='black')
    plt.title(f"Channel {channel_num} (Position: {channel_num * dx:.2f} m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def create_time_file_dict(file_list):
    D = dict()
    for file in file_list:
        t = parse_datetime_from_filename(file)
        D[t] = file
    S = OrderedDict(sorted(D.items(), key=lambda t: t[0]))
    return S

def parse_datetime_from_filename(file):
    # Regex to extract the date and time
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})", file)
    if match:
        date = match.group(1)  # "YYYY-MM-DD"
        time = f"{match.group(2)}:{match.group(3)}:{match.group(4)}"  # "HH:MM:SS"
        parsed_datetime_str = f"{date} {time}"
        FRMAT = '%Y-%m-%d %H:%M:%S'
        parsed_datetime = datetime.strptime(parsed_datetime_str, FRMAT)
        return parsed_datetime
    return None

def downsample_data(data, original_fs, target_fs):
    if target_fs >= original_fs:
        raise ValueError("Target sampling rate must be less than original sampling rate.")

    decimation_factor = int(original_fs / target_fs)
    if original_fs % target_fs != 0:
        raise ValueError("Original fs must be divisible by target fs for integer decimation.")

    data_downsampled = signal.decimate(data, decimation_factor, axis=1, zero_phase=True)
    return data_downsampled

def process_data(data):
    data_detrend = signal.detrend(data, type='linear')
    sos = signal.butter(5, [20,100], 'bandpass', fs=500, output='sos')
    data_filtered = signal.sosfilt(sos, data_detrend)
    return data_filtered