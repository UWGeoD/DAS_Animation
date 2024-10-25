import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from queue import Queue

class DAS:
    def __init__(self, file, select_channels=None) -> None:
        self.file = file
        self.data = None
        self.meta_data = dict()
        self.select_channels = select_channels
        self._get_data()
        self.ani = None

    # To do: need to be able to read different type of das files
    def _get_data(self):
        with h5py.File(self.file, "r") as fp:
            data = fp["Acquisition/Raw[0]/RawData"][...]
            if self.select_channels is not None:
                self.data = self._select_channels(data.T, self.select_channels)
            else:
                self.data = data.T
            self.meta_data['dx'] = fp["Acquisition"].attrs["SpatialSamplingInterval"]
            self.meta_data['fs'] = fs = fp['Acquisition/Raw[0]'].attrs["OutputDataRate"]
            self.meta_data['dt'] = 1.0 / fs
    def plot(self, start_time=None, end_time=None, title=None):
        self.ani = plot_das_data(process_data(self.data), self.select_channels, self.meta_data['dx'], self.meta_data['dt'], 
                      start_time, end_time, title)
        #self.ani
    def _select_channels(self, data, channels):
        return data[channels, :]
        

class MulDAS(DAS):
    def __init__(self, file_list, select_channels=None) -> None:
        self.file_list = file_list
        self.select_channels = select_channels
        self._get_data()
    def _get_data(self):
        data_combined = None
        i = 0 
        for file in self.file_list:
            if self.select_channels is not None:
                DAS_temp = DAS(file, self.select_channels)
            else:
                DAS_temp = DAS(file)
            if i == 0:
                data_combined = DAS_temp.data
                self.meta_data = DAS_temp.meta_data
            else:
                data_temp = DAS_temp.data
                data_combined = np.concatenate([data_combined, data_temp], axis=1)
            i += 1
        self.data = data_combined
    def append(self, file_list):
        pass
        


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
    ani = ax.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", 
               extent=[x[0], x[-1], t[-1], t[0]], interpolation="none", animated=True)
    ax.set_xlabel("Channel Number")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    return ani.get_array()
    #fig.show()

# queue element(DAS obj) with stride parameter 


# should put this to Utilities.py and add more flexibility
# Check what DASpy do for preprocessing
def process_data(data):
    data_detrend = signal.detrend(data, type='linear')
    sos = signal.butter(5, [20,100], 'bandpass', fs=500, output='sos')
    data_filtered = signal.sosfilt(sos, data_detrend)
    return data_filtered

epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)