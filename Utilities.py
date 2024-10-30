# input regex for where to find the date info
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict


epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)

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