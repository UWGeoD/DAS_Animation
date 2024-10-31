# input regex for where to find the date info
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib.animation import FuncAnimation


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

def plot_das_data(data, channels, dx, dt, start_time=None, end_time=None, title=None):
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
    ax.set_xlabel("Channel Number")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    fig.show()

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

def process_data(data):
    data_detrend = signal.detrend(data, type='linear')
    sos = signal.butter(5, [20,100], 'bandpass', fs=500, output='sos')
    data_filtered = signal.sosfilt(sos, data_detrend)
    return data_filtered